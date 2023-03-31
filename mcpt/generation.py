import torch
import warnings
import numpy as np

from typing import *

import mcpt


def convert_prompt_to_ids(
        prompt: str,
        tokenizer: mcpt.tokenization.Tokenizer,
        pinyin_tokenizer: mcpt.tokenization.PinyinTokenizer,
        special_tokens: Dict[str, str],
        use_pinyin: bool = False,
) -> Union[List[int], List[List[int]]]:
    prompt_parts = prompt.split(special_tokens['part-separator'])
    prompt_text_ids = tokenizer.convert_tokens_to_ids([special_tokens['start-token']])
    for prompt_part in prompt_parts:
        prompt_text_ids.extend(tokenizer.convert_string_to_ids(prompt_part))
        prompt_text_ids.append(tokenizer.convert_tokens_to_ids(special_tokens['part-separator']))
    prompt_text_ids = prompt_text_ids[:-1]

    if use_pinyin:
        prompt_pinyin_ids = pinyin_tokenizer.convert_tokens_to_ids([special_tokens['start-token']])
        for prompt_part in prompt_parts:
            prompt_pinyin_ids.extend(pinyin_tokenizer.convert_string_to_ids(prompt_part))
            prompt_pinyin_ids.append(
                pinyin_tokenizer.convert_tokens_to_ids(special_tokens['part-separator'])
            )
        prompt_pinyin_ids = prompt_pinyin_ids[:-1]
        if len(prompt_text_ids) != len(prompt_pinyin_ids):
            warnings.warn(f'`text` has size {len(prompt_text_ids)} and `pinyin` has size {len(prompt_pinyin_ids)}.'
                          f' (most likely due to omitted control characters).'
                          f'The pinyin information is discarded.')
            prompt_pinyin_ids = [0] * len(prompt_text_ids)
        prompt_ids = [prompt_text_ids, prompt_pinyin_ids]
    else:
        prompt_ids = prompt_text_ids

    return prompt_ids


def process_samples(
        samples,
        prompt_ids: List[int],
        end_id: int,
        tokenizer: mcpt.tokenization.Tokenizer,
):
    # Exclude the start token.
    samples = np.asarray(samples.to('cpu'))[:, 1:]
    prompt_ids = prompt_ids[1:]
    for sample in samples:
        if end_id in sample:
            sample = sample[:sample.tolist().index(end_id)]
        sample = sample[len(prompt_ids):]
        text_prompt = tokenizer.convert_ids_to_string(prompt_ids)
        text_generated = tokenizer.convert_ids_to_string(sample)
        yield text_prompt, text_generated


class Sampler:

    def __init__(
            self,
            model_config: Dict[str, Any],
            model,
            end_id: int,
            device: str,
            tokenizer=None,
            pinyin_tokenizer=None,
            use_pinyin: bool = False,
    ):
        self._model_config = model_config
        self._model = model
        self._end_id = end_id
        self._device = device
        self._tokenizer = tokenizer
        self._pinyin_tokenizer = pinyin_tokenizer
        self._use_pinyin = use_pinyin

    @staticmethod
    def _top_k_logits(logits, k):
        if k == 0:
            return logits
        logits_flat = logits.view(-1, logits.size(-1))
        indices = torch.topk(logits_flat, k, dim=-1)[1]
        mask = torch.zeros_like(logits_flat).scatter_(-1, indices, 1)
        mask = mask.view(logits.size())
        logits = logits.masked_fill((mask == 0), -1e10)
        return logits

    @staticmethod
    def _top_p_logits(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        n_remove = sorted_indices_to_remove.sum(dim=-1)
        min_logits = sorted_logits.flip(-1).gather(-1, n_remove.unsqueeze(-1))
        logits[logits < min_logits] = -1e10
        return logits

    def _past_shape(self, batch_size: int) -> List[int]:
        return [
            batch_size,
            self._model_config['n_layer'],
            2,
            self._model_config['n_head'],
            -1,  # n_ctx,
            self._model_config['n_embd'] // self._model_config['n_head'],
        ]

    def _process_logits(self, logits, config: Dict[str, Any], candidates):
        if candidates is None:
            logits = logits / config.get('temperature', 1.0)
            logits = self._top_k_logits(logits, k=config.get('top_k', 1))
            logits = self._top_p_logits(logits, p=config.get('top_p', 1.0))
            log_probs = torch.nn.functional.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            target_logits = logits[:, candidates]
            arg_preds = torch.argmax(target_logits, dim=-1)
            prev = torch.tensor(candidates)[arg_preds].unsqueeze(-1).to(self._device)
        if self._use_pinyin:
            generated_tokens = self._tokenizer.convert_ids_to_tokens(prev.view((-1,)))
            pinyin_ids = self._pinyin_tokenizer.convert_tokenizer_tokens_to_ids(generated_tokens)
            pinyin_ids = pinyin_ids.view((-1, 1))
            prev = torch.cat((prev, pinyin_ids), dim=1).view((-1, 2, 1))
        return prev

    def batch_sample(self, prompt_ids, config: Dict[str, Any], candidates=None):
        batch_size = config.get('batch_size', 1)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int32).to(self._device)
        context = prompt_ids.repeat(batch_size, 1, 1) if self._use_pinyin else prompt_ids.repeat(batch_size, 1)
        past, prev, output = None, context, context
        for i in range(config.get('length')):
            with torch.no_grad():
                logits, presents = self._model(prev, past=past)
            presents = presents.view(self._past_shape(batch_size=batch_size))
            prev = self._process_logits(logits[:, -1], config, candidates)
            past = presents if past is None else torch.cat((past, presents), dim=-2)
            output = torch.cat((output, prev), dim=-1)
            if batch_size == 1 and (prev[0][0].item() if self._use_pinyin else prev.item()) == self._end_id:
                break
        return output[:, 0] if self._use_pinyin else output

    def sample(self, prompt_ids, config: Dict[str, Any], candidates=None):
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int32).to(self._device)
        context = prompt_ids.unsqueeze(0)
        past = None
        prev = context
        for i in range(config.get('length')):
            with torch.no_grad():
                logits, presents = self._model(prev, past=past)
            presents = presents.view(self._past_shape(batch_size=1))
            prev = self._process_logits(logits[:, -1], config, candidates)
            token_id = prev[0][0].item() if self._use_pinyin else prev.item()
            if token_id == self._end_id:
                raise StopIteration
            else:
                yield token_id
            past = presents if past is None else torch.cat((past, presents), dim=-2)
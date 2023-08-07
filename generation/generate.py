import os
import cmd
import fire
import string
import warnings
import importlib

from typing import *

import mcpt


class MCPTGenerate(cmd.Cmd):

    def __init__(
            self,
            generation_config: Dict[str, Any],
            tokenizer: mcpt.Tokenizer,
            pinyin_tokenizer: Optional[mcpt.PinyinTokenizer],
            model: mcpt.Model,
            device: str,
            special_tokens: Dict[str, str],
            prompt: str,
            prefix: str,
            suffix: str,
            plugins: Optional[List[Callable]] = None,
            debug: bool = False,
    ):
        super().__init__()
        self._generation_config = generation_config
        self._tokenizer = tokenizer
        self._pinyin_tokenizer = pinyin_tokenizer
        self._model = model
        self._use_pinyin = self._model.config.get('use_pinyin', False)
        self._special_tokens = special_tokens
        self._prompt = prompt
        self._prefix = prefix
        self._suffix = suffix
        self._plugins = plugins or []
        self._debug = debug
        self._end_id = self._tokenizer.convert_tokens_to_ids(self._special_tokens['end_token'])
        self._renew_cmd_prompt()
        self.use_rawinput = True
        self._sampler = mcpt.generation.Sampler(
            model=self._model,
            end_id=self._end_id,
            device=device,
            tokenizer=self._tokenizer,
            pinyin_tokenizer=self._pinyin_tokenizer,
            verbose=1 if self._debug else 0,
        )

    def _renew_cmd_prompt(self):
        prompt = f'{self._prompt[:10]}...' if len(self._prompt) > 10 else self._prompt
        self.prompt = mcpt.text(f'({prompt})', style=mcpt.ERROR)
        self.prompt += f' [max length: {mcpt.text(self._generation_config["max_length"], style=mcpt.STRUCTURE)}'
        if self._prefix:
            self.prompt += f', prefix: {mcpt.text(self._prefix, style=mcpt.STRUCTURE)}'
        if self._suffix:
            self.prompt += f', suffix: {mcpt.text(self._suffix, style=mcpt.STRUCTURE)}'
        self.prompt += '] -> '

    def _print_samples(self, samples, prompt_ids: List[int]):
        for idx, (_, text_generated) in enumerate(mcpt.generation.process_samples(
                samples=samples,
                prompt_ids=prompt_ids,
                end_id=self._end_id,
                tokenizer=self._tokenizer,
        )):
            text_generated = text_generated.replace(self._special_tokens['new_line'], '\\n')
            print(mcpt.text(f'GENERATED [{idx + 1}]', style=mcpt.WARNING), end=' ')
            print(text_generated)

    def _print_char(self, token_id: int):
        token = self._tokenizer.convert_ids_to_string(token_id)
        if token.startswith('##'):
            token = token[2:]
        elif token[0] in set(list(string.ascii_letters)):
            token = ' ' + token
        elif token == self._special_tokens['new_line']:
            token = '\\n'
        print(token, end='', flush=True)

    def _generate(self):
        step_by_step = self._generation_config['batch_size'] == 1
        backward = self._model.config.get('backward', False)

        print(mcpt.text('QUERY', style=mcpt.WARNING), self._prompt)

        prefix = self._prefix
        for plugin in self._plugins:
            if '{' + plugin.placeholder + '}' in self._prefix:
                plugin_output = plugin(self._prompt)
                if isinstance(plugin_output, Dict):
                    plugin_output, debug_output = plugin_output['text'], plugin_output['debug']
                    print(debug_output)
                    for k, v in debug_output.items():
                        print(mcpt.text(f'PLUGIN {plugin.placeholder} - {k}', style=mcpt.WARNING), v)
                print(mcpt.text(f'PLUGIN {plugin.placeholder}', style=mcpt.WARNING), plugin_output)
                prefix = prefix.replace('{' + plugin.placeholder + '}', plugin_output)
        prompt = prefix + (self._prompt[::-1] if backward else self._prompt) + self._suffix
        if self._debug:
            print(mcpt.text('PROMPT', style=mcpt.WARNING), prompt)

        prompt_ids = mcpt.generation.convert_prompt_to_ids(
            prompt=prompt,
            tokenizer=self._tokenizer,
            pinyin_tokenizer=self._pinyin_tokenizer,
            special_tokens=self._special_tokens,
            use_pinyin=self._use_pinyin,
        )
        try:
            if step_by_step:
                print(mcpt.text(f'GENERATED', style=mcpt.WARNING), end=' ')
                for token_id in self._sampler.sample(prompt_ids=prompt_ids, config=self._generation_config):
                    self._print_char(token_id)
                print()
            else:
                with mcpt.running(f'Generating {self._generation_config["batch_size"]} sample(s)', timer=True):
                    samples = self._sampler.batch_sample(prompt_ids=prompt_ids, config=self._generation_config)
        except KeyboardInterrupt:
            print()

        if not step_by_step:
            self._print_samples(samples, prompt_ids[0] if self._use_pinyin else prompt_ids)

    def do_set(self, arg):
        allowed_keys = {*self._generation_config.keys(), 'prefix', 'suffix'}
        k, v = arg.split()
        if k not in allowed_keys:
            print(mcpt.text(f'`{k}` is not a valid key. Valid keys are: {allowed_keys}', style=mcpt.ERROR))
            return
        if k in ('max_length', 'batch_size', 'top_k'):
            v = int(v)
        elif k in ('temperature', 'top_p'):
            v = float(v)
        elif k == 'prefix':
            self._prefix = v
        elif k == 'suffix':
            self._suffix = v
        if k == 'max_length' and v > self._model.config['n_ctx']:
            v = self._model.config['n_ctx']
            warnings.warn(f'The max generation length cannot be set to {v}. '
                          f'Clipping the length to {self._model.config["n_ctx"]}.')
        if k in self._generation_config:
            self._generation_config[k] = v
        print(mcpt.text(f'`{k}` is set to {v}', style=mcpt.INFO))
        self._renew_cmd_prompt()

    def do_clear(self, arg):
        if len(arg) == 0:
            os.system('cls' if os.name == 'nt' else 'clear')
        else:
            if arg == 'prefix':
                self._prefix = ''
            elif arg == 'suffix':
                self._suffix = ''
            self._renew_cmd_prompt()

    @staticmethod
    def do_exit(_):
        print(mcpt.text('Goodbye', style=mcpt.INFO))
        return True

    def emptyline(self):
        self._generate()

    def default(self, line: str):
        self._prompt = line.strip()
        self._renew_cmd_prompt()
        self._generate()


def main(
        model: str,
        lora_model: str,
        model_config: str,
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: Optional[str] = '../common/vocab/pinyin-1354.txt',
        batch_size: int = 1,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 1.0,
        device: str = 'cuda',
        special_tokens: Optional[Dict[str, str]] = None,
        prompt: str = '齐小明，科学家',
        prefix: str = '',
        suffix: str = '',
        plugins: Optional[List[str]] = None,
        debug: bool = False,
):
    load_model = model
    load_lora_model = lora_model
    generation_config = {
        'batch_size': batch_size,
        'max_length': max_length,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
    }
    special_tokens = {
        'start_token': '[MASK]',
        'end_token': '[CLS]',
        'part_separator': '[unused1]',
        'segment_separator': '[unused2]',
        'new_line': '[SEP]',
        **(special_tokens or {}),
    }
    try:
        with mcpt.running('Loading configs'):
            model_config = mcpt.load_config(model_config)
            tokenizer = mcpt.Tokenizer(vocab)
            pinyin_tokenizer = mcpt.PinyinTokenizer(
                vocab_file=pinyin_vocab,
                fallback=tokenizer,
            ) if model_config.get('use_pinyin', False) else None
            if max_length > model_config['n_ctx']:
                max_length = model_config['n_ctx']
                warnings.warn(f'The max generation length cannot be set to {max_length}. '
                              f'Clipping the length to {model_config["n_ctx"]}.')
            if plugins is not None:
                plugins = [importlib.import_module(plugin).Plugin() for plugin in plugins]

        with mcpt.running('Loading the model', timer=True):
            model = mcpt.Model.from_config(
                config=model_config,
                load_model=load_model,
                load_lora_model=load_lora_model,
                device=device,
            )
            import numpy as np
            old = model.transformer.blocks[0].attn.c_proj.weight.cpu().numpy()
            model.eval()
            new = model.transformer.blocks[0].attn.c_proj.weight.cpu().numpy()
            print(np.all(old == new))
        MCPTGenerate(
            generation_config=generation_config,
            tokenizer=tokenizer,
            pinyin_tokenizer=pinyin_tokenizer,
            model=model,
            device=device,
            special_tokens=special_tokens,
            prompt=prompt,
            prefix=prefix,
            suffix=suffix,
            plugins=plugins,
            debug=debug,
        ).cmdloop()
    except KeyboardInterrupt:
        print(mcpt.text('\nGoodbye', style=mcpt.INFO))


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)

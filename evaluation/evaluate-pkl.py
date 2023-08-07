import fire
import math
import pickle
from mcpt.concurrent import futures
# import concurrent.futures as futures

from typing import *

import mcpt
import mcpt.evaluation


def _split_process(n: int, workers: List[int]) -> List[int]:
    workers = [w / sum(workers) for w in workers]
    split = [math.floor(n * w) for w in workers]
    left = n - sum(split)
    i = 0
    while left > 0:
        split[i % len(split)] += 1
        left -= 1
        i += 1
    return split


def work(
        x,
        y_true,
        candidates: Optional[List[str]],
        config: Dict[str, Any],
        pid: int,
        offset: int,
        tokenizer: mcpt.Tokenizer,
        special_token_ids: Dict[str, int],
        device: str,
        pinyin_tokenizer: Optional[mcpt.PinyinTokenizer] = None,
        callbacks: Optional[List[Callable]] = None,
):
    eval_fn = mcpt.evaluation.get_eval_fn(config.get('evaluation_method', 'generation'))
    model = mcpt.Model.from_config(
        config=config['model_config'],
        load_model=config['model'],
        device=device,
    )
    y_pred = eval_fn(
        x=x,
        model=model,
        offset=offset,
        y_true=y_true,
        config=config,
        tokenizer=tokenizer,
        candidates=candidates,
        pinyin_tokenizer=pinyin_tokenizer,
        special_token_ids=special_token_ids,
        device=device,
        callbacks=callbacks,
    )
    return pid, y_pred


def main(
        dataset: str = 'CMeEE',
        dataset_config: str = 'configs/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: Optional[str] = '../common/vocab/pinyin-1354.txt',
        use_cache: bool = False,
        special_tokens: Optional[Dict[str, str]] = None,
        slicer: Optional[str] = None,
        **kwargs,
):
    # {'dataset': 'CMeEE', 'dataset_config_path': 'configs/local.yaml',
    #  'model_config_path': '/home/lidongwen/lidongwen/linglong-mcpt/common/model-configs/317M-WSZ1024L24.yaml',
    #  'model_config': {'n_ctx': 1024, 'n_layer': 24, 'n_vocab': 13312, 'n_embd': 1024, 'n_head': 16, 'epsilon': 1e-08,
    #                   'embd_dropout': 0.1, 'attn_dropout': 0.1, 'resid_dropout': 0.1, 'mode': 'sparse', 'stride': 128,
    #                   'c': 8}, 'input_path': '/home/lidongwen/lidongwen/dataset/origin-data',
    #  'cache_path': '/home/lidongwen/lidongwen/dataset/fine-tune/', 'vocab': '../common/vocab/char-13312.txt',
    #  'pinyin_vocab': '../common/vocab/pinyin-1354.txt', 'use_cache': False,
    #  'output_path_template': '{dataset}-{model}-{split}-{template_id}-{timestamp}',
    #  'special_tokens': {'start_token': '[MASK]', 'end_token': '[CLS]', 'part_separator': '[unused1]',
    #                     'segment_separator': '[unused2]', 'entity_prefix': '[unused2]', 'entity_postfix': '[unused3]'},
    #  'verbose': 2, 'slicer': None, 'workers': '1:6', 'items_per_process': None, 'device': 'cuda',
    #  'model': '/home/lidongwen/lidongwen/model/PT-CMeEE-317M/models/v1-torch/E1L0.863742.pt',
    #  'evaluation_method': 'generation', 'template_id': 1, 'split': 'dev', 'evaluation_metric': 'entity_metric',
    #  'output_path': 'CMeEE-E1L0.863742-dev-1-1684134351'}

    with open('/home/lidongwen/lidongwen/chinese-gpt/evaluation/CMeEE-E2L0.911203VL1.011414-dev-1-1684152597.pkl',
              'rb') as f:
        config = pickle.load(f)
        y_pred = pickle.load(f)
    print(config)
    config['input_path'] = '/home/lidongwen/lidongwen/dataset/origin-data'
    config['cache_path'] = '/home/lidongwen/lidongwen/dataset/fine-tune/'
    config['pinyin_vocab'] = '../common/vocab/pinyin-1354.txt'
    config['template_id'] = 1
    config['evaluation_metric'] = 'entity_metric'
    config['model_config'] = {'n_ctx': 1024, 'n_layer': 24, 'n_vocab': 13312, 'n_embd': 1024, 'n_head': 16,
                              'epsilon': 1e-08,
                              'embd_dropout': 0.1, 'attn_dropout': 0.1, 'resid_dropout': 0.1, 'mode': 'sparse',
                              'stride': 128,
                              'c': 8}
    config['special_tokens'] = {'start_token': '[MASK]', 'end_token': '[CLS]', 'part_separator': '[unused1]',
                                'segment_separator': '[unused2]', 'entity_prefix': '[unused2]',
                                'entity_postfix': '[unused3]'}
    config['use_cache'] = False
    config['output_path'] = 'CMeEE-E1L0.863742-dev-1-1684134351'
    config['evaluation_metric'] = 'entity_metric_1'
    eval_metric = mcpt.evaluation.get_eval_metric(config.get('evaluation_metric'))
    tokenizer = mcpt.Tokenizer(vocab)
    special_tokens = config.get('special_tokens', {})
    special_token_ids = {

        key: tokenizer.convert_tokens_to_ids(value)
        for (key, value) in special_tokens.items()
    }

    with mcpt.running(f'Loading {dataset} dataset', spinner=use_cache):
        x, y_true, candidates = mcpt.evaluation.load_dataset(config)

    if slicer is not None:
        slicer = slice(*(int(x) if x else None for x in slicer.split(':')))
        x, y_true = x[slicer], y_true[slicer]
    print(eval_metric)
    if eval_metric is not None:
        print(mcpt.text('Calculating evaluation metrics', style=mcpt.INFO))
        result = eval_metric(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            config=config,
            tokenizer=tokenizer,
            output_path=config['output_path'],
            special_token_ids=special_token_ids,
        )
        print(f'{config["evaluation_metric"]}: {mcpt.print_dict(result)}')
    else:
        print(mcpt.text('No evaluation metric is specified.', style=mcpt.WARNING))


if __name__ == '__main__':
    if mcpt.python_version('3.11'):
        raise RuntimeError('This script is not compatible with Python below 3.11.')
    mcpt.init()
    fire.Fire(main)

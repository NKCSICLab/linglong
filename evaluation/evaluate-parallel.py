import fire
import math
import pickle
import concurrent.futures as futures

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
        load_model=config['model']['checkpoint'],
        load_lora_model=config['model_lora']['checkpoint'],
        device=device,
    )
    model.eval()
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
        dataset: str,
        input_path: str,
        cache_path: str,
        workers: str,
        dataset_config: str = 'configs/local.yaml',
        vocab: str = '../common/vocab/char-13312.txt',
        pinyin_vocab: Optional[str] = '../common/vocab/pinyin-1354.txt',
        use_cache: bool = False,
        output_path_template: str = '{dataset}-{model}-{split}-{template_id}-{timestamp}',
        special_tokens: Optional[Dict[str, str]] = None,
        verbose: int = 2,
        slicer: Optional[str] = None,
        items_per_process: Optional[int] = None,
        device: str = 'cuda',
        **kwargs,
):
    with mcpt.running('Loading configs') as spinner:
        special_tokens = {
            'start_token': '[MASK]',
            'end_token': '[CLS]',
            'part_separator': '[unused1]',
            'segment_separator': '[unused2]',
            **(special_tokens or {}),
        }
        config = mcpt.merge_configs(
            {
                'dataset': dataset,
                'dataset_config_path': dataset_config,
                'input_path': input_path,
                'cache_path': cache_path,
                'vocab': vocab,
                'pinyin_vocab': pinyin_vocab,
                'use_cache': use_cache,
                'output_path_template': output_path_template,
                'special_tokens': special_tokens,
                'verbose': verbose,
                'slicer': slicer,
                'workers': workers,
                'items_per_process': items_per_process,
                'device': device,
            },
            mcpt.load_config(dataset_config, key=dataset),
            kwargs,
        )
        config['model_config'] = mcpt.load_config(config['model']['config'])

        tokenizer = mcpt.Tokenizer(vocab)
        pinyin_tokenizer = mcpt.PinyinTokenizer(vocab_file=pinyin_vocab, fallback=tokenizer) if pinyin_vocab else None
        output_path = mcpt.evaluation.get_output_path(config)
        config['output_path'] = output_path
        special_token_ids = {
            key: tokenizer.convert_tokens_to_ids(value) for (key, value) in special_tokens.items()
        }
        eval_metric = mcpt.evaluation.get_eval_metric(config.get('evaluation_metric'))
        spinner.write(mcpt.pprint(config, export=True))

    with mcpt.running(f'Loading {dataset} dataset', spinner=use_cache):
        x, y_true, candidates = mcpt.evaluation.load_dataset(config)
        if slicer is not None:
            slicer = slice(*(int(x) if x else None for x in slicer.split(':')))
            x, y_true = x[slicer], y_true[slicer]
        candidates = tokenizer.convert_tokens_to_ids(candidates) if candidates is not None else None

    with mcpt.running(f'Setting up workers') as spinner:
        worker_config = {}
        workers = str(workers).split(',')
        n_workers = 0
        for worker in workers:
            if ':' in worker:
                gpu_id, gpu_workers = worker.split(':')
                worker_config[int(gpu_id)] = {'workers': int(gpu_workers), 'processes': 0}
                n_workers += int(gpu_workers)
            else:
                worker_config[0] = {'workers': int(worker), 'processes': 0}
                n_workers += int(worker)
        worker_config = [[k, v] for k, v in worker_config.items()]
        if items_per_process is None:
            n_process = n_workers
            items_per_process = math.ceil(len(x) / n_process)
        else:
            n_process = math.ceil(len(x) / items_per_process)
        process_split = _split_process(n_process, [worker[1]['workers'] for worker in worker_config])
        for processes, worker in zip(process_split, worker_config):
            worker[1]['processes'] = processes
        worker_config = {k: v for k, v in worker_config}
        spinner.write(mcpt.pprint(worker_config, export=True))

    print(mcpt.text('Evaluating', style=mcpt.INFO))
    pid = 0
    y_pred = []
    futures_ = []
    executors = []
    manager = mcpt.evaluation.CallbackManager()
    manager.start()
    lock = manager.Lock()
    # noinspection PyUnresolvedReferences
    progress_bar_callback = manager.ProgressBarCallback(len(x), lock=lock)
    # noinspection PyUnresolvedReferences
    log_result_callback = manager.LogResultCallback(
        path=output_path,
        verbose=verbose,
        tokenizer=tokenizer,
        use_perplexity=config.get('use_perplexity', False),
        use_pinyin=config['model_config'].get('use_pinyin', False),
        lock=lock,
    )
    for gpu_id, worker in worker_config.items():
        device = f'cuda:{gpu_id}' if gpu_id > -1 else 'cpu'
        # noinspection PyArgumentList
        executor = futures.ProcessPoolExecutor(
            max_workers=worker['workers'],
            max_tasks_per_child=1,
        )
        executors.append(executor)
        for _ in range(worker['processes']):
            if len(x[pid * items_per_process: (pid + 1) * items_per_process]) > 0:
                futures_.append(executor.submit(
                    work,
                    pid=pid,
                    config=config,
                    tokenizer=tokenizer,
                    candidates=candidates,
                    offset=pid * items_per_process,
                    pinyin_tokenizer=pinyin_tokenizer,
                    special_token_ids=special_token_ids,
                    x=x[pid * items_per_process: (pid + 1) * items_per_process],
                    y_true=y_true[pid * items_per_process: (pid + 1) * items_per_process],
                    device=device,
                    callbacks=[
                        progress_bar_callback,
                        log_result_callback,
                    ],
                ))
                pid += 1
    for future in futures.as_completed(futures_):
        y_pred.append(future.result())
    progress_bar_callback.close()
    manager.shutdown()
    for executor in executors:
        executor.shutdown()
    y_pred.sort(key=lambda _: _[0])
    y_pred = [b for _, a in y_pred for b in a]

    if verbose > 0:
        with mcpt.running('Saving intermediate results'):
            with open(output_path + '.pkl', 'wb') as f:
                pickle.dump(config, f)
                pickle.dump(y_pred, f)

    if eval_metric is not None:
        print(mcpt.text('Calculating evaluation metrics', style=mcpt.INFO))
        result = eval_metric(
            x=x,
            y_true=y_true,
            y_pred=y_pred,
            config=config,
            tokenizer=tokenizer,
            output_path=output_path,
            special_token_ids=special_token_ids,
        )
        print(f'{config["evaluation_metric"]}: {mcpt.pprint(result, export=True)}')
    else:
        print(mcpt.text('No evaluation metric is specified.', style=mcpt.WARNING))


if __name__ == '__main__':
    if not mcpt.python_version('3.11'):
        raise RuntimeError('This script is not compatible with Python below 3.11.')
    mcpt.init()
    fire.Fire(main)

import fire
import torch
import pathlib
import contextlib

from typing import *

import mcpt
import mcpt.records
import loralib as lora

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    hvd = mcpt.stubs.Horovod()


def main(
        training_data: str,
        model_config: str,
        training_config: str,
        override_config: Optional[Dict] = None,
        training_meta: str = 'train-meta.json',
        validation_data: Optional[str] = None,
        validation_meta: str = 'valid-meta.json',
        epochs: int = 20,
        load_model: Optional[str] = None,
        save_path: str = './ckpt',
        save_frequency: Union[int, str, List[Union[int, str]]] = 'epoch',
        log_frequency: int = 10,
        device: str = 'cuda',
):
    hvd.init()
    mcpt.bind_gpu(hvd)

    with mcpt.running('Loading configs', hvd=hvd) as spinner:
        model_config_path, training_config_path = model_config, training_config
        model_config = mcpt.load_config(model_config_path)
        model_config = mcpt.merge_configs(model_config, (override_config or {}).get('model_config', {}))
        training_config = mcpt.load_config(training_config_path)
        training_config = mcpt.merge_configs(training_config, (override_config or {}).get('training_config', {}))
        config = {
            'training_data': training_data,
            'model_config_path': model_config_path,
            'training_config_path': training_config_path,
            'override_config': override_config,
            'training_meta': training_meta,
            'validation_data': validation_data,
            'validation_meta': validation_meta,
            'epochs': epochs,
            'load_model': load_model,
            'save_path': save_path,
            'save_frequency': save_frequency,
            'log_frequency': log_frequency,
            'device': device,
            'model_config': model_config,
            'training_config': training_config,
        }
        save_path = pathlib.Path(save_path)
        spinner.write(mcpt.pprint(config, export=True))

    with mcpt.running('Loading the dataset', hvd=hvd, timer=True):
        train_loader = mcpt.records.load(
            path=training_data,
            meta=training_meta,
            batch_size=training_config['train_micro_batch_size_per_gpu'],
            dp_size=hvd.size(),
            dp_rank=hvd.rank(),
            use_pinyin=model_config.get('use_pinyin', False),
        )
        validation_loader = mcpt.records.load(
            path=validation_data,
            meta=validation_meta,
            batch_size=training_config['train_micro_batch_size_per_gpu'],
            dp_size=hvd.size(),
            dp_rank=hvd.rank(),
            use_pinyin=model_config.get('use_pinyin', False),
        )

    with mcpt.running('Loading the model', hvd=hvd, timer=True):
        model = mcpt.Model.from_config(
            config=model_config,
            load_model=load_model,
            device=device,
        )
        training_config['optimizer']['params']['lr'] = \
            training_config['optimizer']['params']['lr'] * hvd.size() * training_config['gradient_accumulation_steps']
        optimizer = mcpt.train.optimizers.adamw(model.parameters(), config=training_config['optimizer']['params'])
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            sparse_as_dense=True,
            backward_passes_per_step=training_config['gradient_accumulation_steps'],
        )
        lr_scheduler = mcpt.train.schedulers.cosine_annealing_warmup(
            optimizer,
            config=training_config,
            n_ctx=model.config['n_ctx'],
            dp_size=hvd.size(),
        )
        # hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        callbacks = []
        if hvd.rank() == 0:
            if not isinstance(save_frequency, Union[Tuple, List]):
                save_frequency = (save_frequency,)
            for freq in save_frequency:
                if isinstance(freq, str):
                    assert freq in ('epoch', 'step')
                callbacks.append(
                    mcpt.train.callbacks.ModelCheckpointCallback(
                        save_path=save_path,
                        save_frequency=freq,
                    ),
                )

    if hvd.rank() == 0:
        print(model)
    lora.mark_only_lora_as_trainable(model, 'all')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")
    scaler = torch.cuda.amp.GradScaler() if training_config['fp16']['enabled'] else mcpt.stubs.Noop()
    for epoch in range(1, epochs + 1):
        if hvd.rank() == 0:
            print(f'Epoch {epoch}/{epochs}')
        train_tqdm = mcpt.tqdm(enumerate(train_loader), total=len(train_loader), hvd=hvd)
        loss = 0
        for batch_idx, (data, target) in train_tqdm:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            with (torch.cuda.amp.autocast() if training_config['fp16']['enabled'] else contextlib.suppress()):
                logits = model(data)['logits']
                loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), target, ignore_index=0)
            if training_config['fp16']['enabled']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if not isinstance(hvd, mcpt.stubs.Horovod):
                optimizer.synchronize()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config['gradient_clipping'])
            with (contextlib.suppress() if isinstance(hvd, mcpt.stubs.Horovod) else optimizer.skip_synchronize()):
                if training_config['fp16']['enabled']:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            scaler.update()
            lr_scheduler.step()
            loss = loss.item()
            if hvd.rank() == 0 and (batch_idx + 1) % log_frequency == 0:
                train_tqdm.write(
                    f'Train Epoch: {epoch}/{epochs} [{batch_idx + 1}/{len(train_loader)}] '
                    f'Loss: {loss}' +
                    (f' Loss Scale: {scaler.get_scale()}' if training_config['fp16']['enabled'] else ''),
                )
            for callback in callbacks:
                callback(model=model, epoch=epoch, batch=batch_idx + 1, loss=loss)
        if validation_data is not None:
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(validation_loader):
                    data, target = data.to(device), target.to(device)
                    logits = model(data)['logits']
                    val_loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), target, ignore_index=0)
                    if hvd.rank() == 0:
                        print(f'Valid Epoch: [{batch_idx + 1}/{len(validation_loader)}] Loss: {val_loss.item()}')
            model.train()
        for callback in callbacks:
            callback(
                model=model,
                epoch=epoch,
                loss=loss,
                val_loss=val_loss.item() if validation_data is not None else None,
                end_of_epoch=True,
            )


if __name__ == '__main__':
    mcpt.init()
    fire.Fire(main)

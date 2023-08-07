import torch
import pathlib
import loralib as lora

from typing import *

import mcpt


class ModelCheckpointCallback:

    def __init__(
            self,
            save_path: pathlib.Path,
            save_frequency: Union[int, str],
    ):
        self.save_path = save_path
        self.save_frequency = save_frequency
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _get_save_name(self, batch: Optional[int] = None, val_loss: Optional[float] = None,) -> str:
        save_name = 'E{epoch}'
        if self.save_frequency != 'epoch' and batch is not None:
            save_name += 'B{batch}'
        save_name += 'L{loss:.6f}'
        if self.save_frequency == 'epoch' and val_loss is not None:
            save_name += 'VL{val_loss:.6f}'
        save_name += '.pt'
        return save_name

    def __call__(
            self,
            model: mcpt.Model,
            epoch: int,
            loss: float,
            batch: Optional[int] = None,
            val_loss: Optional[float] = None,
            end_of_epoch: bool = False,
            prefix: Optional[str] = None,
    ):
        save_name = self._get_save_name(batch=batch, val_loss=val_loss)
        if prefix is not None:
            save_name = prefix + save_name
        save_name = save_name.format(epoch=epoch, batch=batch, loss=loss, val_loss=val_loss)
        if self.save_frequency == 'epoch' and not end_of_epoch:
            return
        if isinstance(self.save_frequency, int):
            if end_of_epoch or batch % self.save_frequency != 0:
                return
        torch.save(lora.lora_state_dict(model), str(self.save_path / save_name))

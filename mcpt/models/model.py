import torch
import torch.nn as nn

from typing import *


class Model(nn.Module):

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.config = self.transformer.config

    def forward(self, inputs, past=None):
        h = self.transformer(inputs, past=past)
        h, present = h['hidden_states'], h['present']
        logits = torch.matmul(h, self.transformer.wte.weight.t())
        return {
            'logits': logits,
            'present': present,
        }

    def hidden_states(self, inputs, past=None):
        return self.transformer(inputs, past=past)['hidden_states']

    @classmethod
    def from_config(
            cls,
            config: Union[str, Dict[str, Any]],
            load_model: Optional[str] = None,
            load_lora_model: Optional[str] = None,
            device: Optional = None,
            strict: bool = True,
    ) -> 'Self':
        import mcpt
        if isinstance(config, str):
            config = mcpt.load_config(config)
        if config.get('use_pinyin', False):
            model = cls(mcpt.models.MCPTPinyinModel(config))
        else:
            model = cls(mcpt.models.MCPTModel(config))
        if load_model is not None:
            model.load_state_dict(torch.load(load_model, map_location=device), strict=False)
        if load_lora_model is not None:
            model.load_state_dict(torch.load(load_lora_model, map_location=device), strict=False)
        # print(model.transformer.wte.weight)
        # print(model.transformer.blocks[0].attn.c_attn.weight)
        # for name, param in model.transformer.blocks[0].attn.c_attn.named_parameters():
        #     print(name)
        #     print(param.shape)
        #     print(param)
        # # print(model.transformer.blocks[0].attn.c_attn.w)
        #
        # input()
        if device is not None:
            model.to(device)
        return model


class RewardModel(Model):

    def __init__(self, transformer):
        super().__init__(transformer)
        self.reward_head = nn.Linear(self.config['n_embd'], 1, bias=False)

    def forward(self, inputs, past=None):
        h = self.hidden_states(inputs, past=past)
        rewards = self.reward_head(h).squeeze(-1)
        return rewards

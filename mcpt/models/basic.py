import math
import torch
import torch.nn as nn
import loralib as lora

from typing import *


class Conv1D(nn.Module):

    def __init__(self, units: int, n_input: int, init_std: float = 0.02):
        super().__init__()
        self.units = units
        self.n_input = n_input
        w = torch.empty(self.n_input, self.units)
        nn.init.normal_(w, std=init_std)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(self.units))

    def forward(self, inputs):
        start = list(inputs.shape)[:-1]
        inputs_flat = inputs.view(-1, self.n_input)
        w_flat = self.w.view(-1, self.units)
        h = torch.addmm(self.b, inputs_flat, w_flat)
        h = h.view(start + [self.units])
        return h


class MLP(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # self.c_fc = Conv1D(
        #     units=config['n_embd'] * 4,
        #     n_input=config['n_embd'],
        # )
        # self.c_proj = Conv1D(
        #     units=config['n_embd'],
        #     n_input=config['n_embd'] * 4,
        #     init_std=0.02 * (1.0 / math.sqrt(2 * config['n_layer'])),
        # )
        self.c_fc = lora.Linear(in_features=config['n_embd'], out_features=config['n_embd'] * 4, r=8,
                                fan_in_fan_out=True)
        self.c_proj = lora.Linear(in_features=config['n_embd'] * 4, out_features=config['n_embd'], r=8,
                                  fan_in_fan_out=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config['resid_dropout'])

    def forward(self, inputs):
        h = self.c_fc(inputs)
        h = self.act(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h


class Attention(nn.Module):

    def __init__(self, config: Dict[str, Any], blk_idx: int):
        super().__init__()
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.mode = config['mode']
        self.blk_idx = blk_idx
        self.stride = config.get('stride')
        self.c = config.get('c')
        # self.c_attn = Conv1D(
        #     units=3 * self.n_embd,
        #     n_input=self.n_embd,
        # )
        self.c_attn = lora.MergedLinear(
            self.n_embd, self.n_embd * 3,
            r=config['lora_attn_dim'],
            lora_alpha=config['lora_attn_alpha'],
            lora_dropout=config['lora_dropout'],
            enable_lora=[True, True, True],
            fan_in_fan_out=True,
            merge_weights=False
        )
        self.c_proj = lora.Linear(self.n_embd, self.n_embd, r=8, fan_in_fan_out=True)
        # self.c_proj = Conv1D(
        #     units=self.n_embd,
        #     n_input=self.n_embd,
        #     init_std=0.02 * (1.0 / math.sqrt(2 * config['n_layer'])),
        # )
        self.attn_mask = None
        self.dropout = nn.Dropout(config['attn_dropout'])

    def _attn(self, q, k, v):
        w = torch.matmul(q, k.transpose(-1, -2))
        w = w / torch.tensor(v.size(-1) ** 0.5, dtype=w.dtype, device=w.device)
        w = self._mask_attn_weights(w)
        w = nn.functional.softmax(w, dim=-1)
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        w = w.type(v.dtype)
        h = torch.matmul(w, v)
        return h

    @staticmethod
    def _attention_mask(nd, ns, *, device, dtype):
        return torch.tril(torch.ones([nd, ns], dtype=dtype, device=device), diagonal=ns - nd)

    @staticmethod
    def _sparse_attention_mask(nd, ns, stride, c, *, device, dtype):
        layout = torch.zeros([ns, ns], dtype=dtype, device=device)
        for idx in range(c):
            layout[:, (stride - 1 - idx)::stride] = 1
        for q_idx in range(ns):
            row = q_idx // stride
            layout[q_idx, row * stride:(row + 1) * stride] = 1
            # Any query cannot attend to keys above it.
            layout[q_idx, q_idx + 1:] = 0
        return layout[(ns - nd):]

    def _mask_attn_weights(self, w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = w.shape
        if self.attn_mask is None or self.attn_mask.size() != torch.Size([nd, ns]):
            if self.mode == 'sparse' and self.blk_idx % 2 != 0:
                self.attn_mask = self._sparse_attention_mask(
                    nd, ns, self.stride, self.c, device=w.device, dtype=w.dtype
                )
            else:
                self.attn_mask = self._attention_mask(nd, ns, device=w.device, dtype=w.dtype)
        b = self.attn_mask.view(1, 1, nd, ns)
        w = w * b - 1e10 * (1 - b)
        return w

    def _split_heads(self, x):
        *start, m = x.size()
        x = x.view(start + [self.n_head, m // self.n_head])
        x = x.permute(0, 2, 1, 3)
        return x

    @staticmethod
    def _merge_heads(x):
        x = x.permute([0, 2, 1, 3]).contiguous()
        *start, a, b = x.size()
        x = x.view(start + [a * b])
        return x

    def forward(self, inputs, past=None):
        q, k, v = self.c_attn(inputs).split(self.n_embd, dim=2)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        present = torch.stack([k, v], dim=1)
        if past is not None:
            pk, pv = torch.unbind(past, dim=1)
            k = torch.cat((pk, k), dim=-2)
            v = torch.cat((pv, v), dim=-2)
        h = self._attn(q, k, v)
        h = self._merge_heads(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return h, present


class Block(nn.Module):

    def __init__(self, config: Dict[str, Any], blk_idx: int):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config['n_embd'], eps=config['epsilon'])
        self.attn = Attention(config, blk_idx=blk_idx)
        self.ln_2 = nn.LayerNorm(config['n_embd'], eps=config['epsilon'])
        self.mlp = MLP(config)

    def forward(self, inputs, past=None):
        h_1 = self.ln_1(inputs)
        h_1, present = self.attn(h_1, past=past)
        h_1 = h_1 + inputs
        h_2 = self.ln_2(h_1)
        h_2 = self.mlp(h_2)
        h = h_2 + h_1
        return h, present


class MCPTModel(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.n_embd = config['n_embd']
        # self.wte = nn.Embedding(config['n_vocab'], self.n_embd)
        # self.wpe = nn.Embedding(config['n_ctx'], self.n_embd)
        self.wte = lora.Embedding(num_embeddings=config['n_vocab'], embedding_dim=self.n_embd, r=8)
        self.wpe = lora.Embedding(num_embeddings=config['n_ctx'], embedding_dim=self.n_embd, r=8)
        self.drop = nn.Dropout(config['embd_dropout'])
        self.blocks = nn.ModuleList([Block(config, blk_idx=i) for i in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(self.n_embd, eps=config['epsilon'])

    def forward(self, inputs, past=None):
        if past is None:
            past_length = 0
            pasts = tuple([None] * len(self.blocks))
        else:
            past_length = past.size(-2)
            pasts = torch.unbind(past, dim=1)
        position_ids = torch.arange(past_length, inputs.size(-1) + past_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).view(-1, inputs.size(-1))
        inputs_embd = self.wte(inputs)
        position_embd = self.wpe(position_ids)
        h = inputs_embd + position_embd
        h = self.drop(h)
        presents = []
        for blk_idx, past in enumerate(pasts):
            h, present = self.blocks[blk_idx](h, past)
            presents.append(present)
        present = torch.stack(presents, dim=1)
        h = self.ln_f(h)
        return {
            'hidden_states': h,
            'present': present,
        }

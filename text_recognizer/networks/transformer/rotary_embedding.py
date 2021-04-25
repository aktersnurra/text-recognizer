"""Roatary embedding.

Stolen from lucidrains:
    https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

Explanation of roatary:
    https://blog.eleuther.ai/rotary-embeddings/

"""
from typing import Tuple

from einops import rearrange
import torch
from torch import nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor, seq_dim: int = 1) -> Tensor:
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, freqs: Tensor) -> Tuple[Tensor, Tensor]:
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k

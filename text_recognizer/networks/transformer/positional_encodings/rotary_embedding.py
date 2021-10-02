"""Roatary embedding.

Stolen from lucidrains:
    https://github.com/lucidrains/rotary-embedding-torch

Explanation of roatary:
    https://blog.eleuther.ai/rotary-embeddings/
"""
from typing import Tuple

from einops import rearrange
import torch
from torch import nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor, seq_dim: int = 1) -> Tensor:
        """Encodes tensor x with rotary embeddings."""
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, "n d -> () () n d")


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

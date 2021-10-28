"""Roatary embedding.

Stolen from lucidrains:
    https://github.com/lucidrains/rotary-embedding-torch

Explanation of roatary:
    https://blog.eleuther.ai/rotary-embeddings/
"""
import torch
from torch import nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes tensor x with rotary embeddings."""
        freqs = self.inv_freqs
        freqs = torch.einsum("..., f -> ... f", x.type(freqs.dtype), freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


def rotate_half(x: Tensor) -> Tensor:
    x = x.reshape((x.shape[0], -1, 2, x.shape[-1] // 2))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

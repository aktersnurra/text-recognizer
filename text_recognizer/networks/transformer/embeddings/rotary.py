"""Roatary embedding.

Stolen from lucidrains:
    https://github.com/lucidrains/rotary-embedding-torch

Explanation of roatary:
    https://blog.eleuther.ai/rotary-embeddings/
"""
import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes tensor x with rotary embeddings."""
        n = x.shape[-2]
        t = torch.arange(n, device=x.device).type_as(self.inv_freqs)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]


def rotate_half(x: Tensor) -> Tensor:
    if len(x.shape) == 3:
        x = x.reshape((x.shape[0], -1, 2, x.shape[-1] // 2))
    else:
        x = x.reshape((x.shape[0], x.shape[1], -1, 2, x.shape[-1] // 2))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

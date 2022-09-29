"""Roatary embedding.

Stolen from lucidrains:
    https://github.com/lucidrains/rotary-embedding-torch

Explanation of roatary:
    https://blog.eleuther.ai/rotary-embeddings/
"""
from inspect import isfunction

from einops import rearrange, repeat
import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freqs", inv_freqs)
        self.cache = {}

    def rotate(self, t: Tensor, dim: int = -2) -> Tensor:
        """Rotate vector."""
        device, n = t.device, t.shape[dim]
        freqs = self.forward(lambda: torch.arange(n, device=device), cache_key=n)
        return apply_rotary_emb(t, freqs)

    def forward(self, t: Tensor, cache_key: int) -> Tensor:
        """Encodes tensor x with rotary embeddings."""
        if cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.inv_freqs
        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        self.cache[cache_key] = freqs
        return freqs


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(t: Tensor, freqs: Tensor, start_index: int = 0) -> Tensor:
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], (
        f"feature dimension {t.shape[-1]} is not of sufficient size to rotate"
        f"in all the positions {rot_dim}"
    )
    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim=-1)

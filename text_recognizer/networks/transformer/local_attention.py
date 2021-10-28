"""Local attention module.

Also stolen from lucidrains from here:
https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py
"""
from functools import reduce
from operator import mul
from typing import Optional, Tuple

import attr
from einops import rearrange
import torch
from torch import einsum
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from text_recognizer.networks.transformer.attention import apply_rotary_emb


@attr.s(eq=False)
class LocalAttention(nn.Module):
    """Local windowed attention."""

    dim: int = attr.ib()
    num_heads: int = attr.ib()
    dim_head: int = attr.ib(default=64)
    window_size: int = attr.ib(default=128)
    look_back: int = attr.ib(default=1)
    dropout_rate: float = attr.ib(default=0.0)

    def __attrs_pre_init__(self) -> None:
        """Pre init constructor."""
        super().__init__()

    def __attrs_post_init__(self) -> None:
        """Post init constructor."""
        self.scale = self.dim ** -0.5
        inner_dim = self.dim * self.dim_head

        self.query = nn.Linear(self.dim, inner_dim, bias=False)
        self.key = nn.Linear(self.dim, inner_dim, bias=False)
        self.value = nn.Linear(self.dim, inner_dim, bias=False)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Feedforward
        self.fc = nn.Linear(inner_dim, self.dim)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Computes windowed attention."""
        b, n, d = x.shape
        if not n % self.window_size:
            RuntimeError(
                f"Sequence length {n} must be divisable with window size {self.window_size}"
            )

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )
        q, k, v = (
            apply_rotary_emb(q, k, v, rotary_pos_emb)
            if rotary_pos_emb is not None
            else (q, k, v,)
        )

        num_windows = n // self.window_size

        # Compute buckets
        b_n = (
            torch.arange(self.num_heads * n)
            .type_as(q)
            .reshape(1, self.num_heads, num_windows, self.window_size)
        )
        bq, bk, bv = map(
            lambda t: t.reshape(b, self.num_heads, num_windows, self.window_size, -1),
            (q, k, v),
        )

        bk = look_around(bk, backward=self.look_back)
        bv = look_around(bv, backward=self.look_back)
        bq_k = look_around(b_n, backward=self.look_back)

        # Compute the attention.
        energy = einsum("b h n i d, b h n j d -> b h n i j", bq, bk) * self.scale
        mask_value = -torch.finfo(energy.dtype).max

        # Causal mask.
        causal_mask = b_n[:, :, :, :, None] < bq_k[:, :, :, None, :]
        energy = energy.masked_fill_(causal_mask, mask_value)
        del causal_mask

        bucket_mask = bq_k[:, :, :, None, :] == -1
        energy.masked_fill_(bucket_mask, mask_value)
        del bucket_mask

        energy = apply_input_mask(
            b,
            energy=energy,
            mask=mask,
            backward=self.look_back,
            window_size=self.window_size,
            num_windows=num_windows,
            num_heads=self.num_heads,
            mask_value=mask_value,
        )

        attn = F.softmax(energy, dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h n i j, b h n j d -> b h n i d", attn, bv)
        out = out.reshape(-1, n, d)

        out = self.fc(out)
        return out, attn


def merge_dims(ind_from: int, ind_to: int, tensor: Tensor) -> Tensor:
    """Merge dimensions."""
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t: Tensor, dim: int, k: int, unsqueeze: bool = True) -> Tensor:
    """Expand tensors dimensions."""
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def look_around(x: Tensor, backward: int, pad_value: int = -1, dim: int = 3) -> Tensor:
    """Apply windowing."""
    n = x.shape[2]
    dims = (len(x.shape) - dim) * (0, 0)
    x_pad = F.pad(x, (*dims, backward, 0), value=pad_value)
    tensors = [x_pad[:, :, ind : (ind + n), ...] for ind in range(backward + 1)]
    return torch.cat(tensors, dim=dim)


def apply_input_mask(
    b: int,
    energy: Tensor,
    mask: Tensor,
    backward: int,
    window_size: int,
    num_windows: int,
    num_heads: int,
    mask_value: Tensor,
) -> Tensor:
    """Applies input mask to energy tensor."""
    h = b // mask.shape[0]
    mask = torch.cat([mask] * num_heads)
    mask = mask.reshape(-1, num_heads, num_windows, window_size)
    mq = mk = mask
    mk = look_around(mk, pad_value=False, backward=backward)
    mask = mq[:, :, :, :, None] * mk[:, :, :, None, :]
    mask = merge_dims(1, 2, expand_dim(mask, 2, h))
    energy.masked_fill_(~mask, mask_value)
    del mask
    return energy

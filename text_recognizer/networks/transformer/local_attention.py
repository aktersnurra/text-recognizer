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
    dim: int = attr.ib()
    dim_head: int = attr.ib(default=64)
    window_size: int = attr.ib(default=128)
    look_back: int = attr.ib(default=1)
    dropout_rate: float = attr.ib(default=0.0)

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
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
        b, n, _, device, dtype = *x.shape, x.device, x.dtype
        assert (
            n % self.window_size
        ), f"Sequence length {n} must be divisable with window size {self.window_size}"

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
        b_n = torch.arange(n).type_as(q).reshape(1, num_windows, self.window_size)
        bq, bk, bv = map(
            lambda t: t.reshape(b, num_windows, self.window_size, -1), (q, k, v)
        )

        bk = look_around(bk, backward=self.backward)
        bv = look_around(bv, backward=self.backward)
        bq_k = look_around(b_n, backward=self.backward)

        # Compute the attention.
        energy = einsum("b h i d, b h j d -> b h i j", bq, bk) * self.scale
        mask_value = -torch.finfo(energy.dtype).max

        # Causal mask.
        causal_mask = b_n[:, :, :, None] < bq_k[:, :, None, :]
        energy = energy.masked_fill_(causal_mask, mask_value)
        del causal_mask

        bucket_mask = bq_k[:, :, None, :] == -1
        energy.masked_fill_(bucket_mask, mask_value)
        del bucket_mask

        energy = apply_input_mask(
            b,
            energy=energy,
            mask=mask,
            backward=self.backward,
            window_size=self.window_size,
            num_windows=num_windows,
            mask_value=mask_value,
        )

        attn = F.softmax(energy, dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, bv)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.fc(out)
        return out, attn


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def look_around(x: Tensor, backward: int, pad_value: int = -1, dim: int = 2) -> Tensor:
    n = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    x_pad = F.pad(x, (*dims, backward, 0), value=pad_value)
    tensors = [x_pad[:, ind : (ind + n), ...] for ind in range(backward + 1)]
    return torch.cat(tensors, dim=dim)


def apply_input_mask(
    b: int,
    energy: Tensor,
    mask: Tensor,
    backward: int,
    window_size: int,
    num_windows: int,
    mask_value: Tensor,
) -> Tensor:
    h = b // mask.shape[0]
    mask = mask.reshape(-1, window_size, num_windows)
    mq = mk = mask
    mk = look_around(mk, pad_value=False, backward=backward)
    mask = mq[:, :, :, None] * mk[:, :, None, :]
    mask = merge_dims(0, 1, expand_dim(mask, 1, h))
    energy.masked_fill_(~mask, mask_value)
    del mask
    return energy

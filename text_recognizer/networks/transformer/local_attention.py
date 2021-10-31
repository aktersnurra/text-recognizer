"""Local attention module.

Also stolen from lucidrains from here:
https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py
"""
from functools import reduce
from operator import mul
from typing import Optional, Tuple

import attr
import torch
from torch import einsum
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from text_recognizer.networks.transformer.embeddings.rotary import (
    RotaryEmbedding,
    rotate_half,
)


@attr.s(eq=False)
class LocalAttention(nn.Module):
    """Local windowed attention."""

    dim: int = attr.ib()
    num_heads: int = attr.ib()
    dim_head: int = attr.ib(default=64)
    window_size: int = attr.ib(default=128)
    look_back: int = attr.ib(default=1)
    dropout_rate: float = attr.ib(default=0.0)
    rotary_embedding: Optional[RotaryEmbedding] = attr.ib(default=None)

    def __attrs_pre_init__(self) -> None:
        """Pre init constructor."""
        super().__init__()

    def __attrs_post_init__(self) -> None:
        """Post init constructor."""
        self.scale = self.dim ** -0.5
        inner_dim = self.num_heads * self.dim_head

        self.to_qkv = nn.Linear(self.dim, 3 * inner_dim, bias=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Feedforward
        self.fc = nn.Linear(inner_dim, self.dim)

    def _to_embeddings(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert input into query, key, and value."""

        def _split_heads(t: Tensor) -> Tensor:
            return _reshape_dim(t, -1, (-1, self.dim_head)).transpose(1, 2).contiguous()

        def _merge_into_batch(t: Tensor) -> Tensor:
            return t.reshape(-1, *t.shape[-2:])

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(_split_heads, qkv)

        q, k, v = map(_merge_into_batch, (q, k, v))

        if self.rotary_embedding is not None:
            embedding = self.rotary_embedding(q)
            q, k = _apply_rotary_emb(q, k, embedding)
        return q, k, v

    def _create_buckets(
        self, q: Tensor, k: Tensor, v: Tensor, n: int, b: int, num_windows: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        b_n = torch.arange(n).type_as(q).reshape(1, num_windows, self.window_size)
        bq, bk, bv = map(
            lambda t: t.reshape(b, num_windows, self.window_size, -1), (q, k, v),
        )

        bk = look_around(bk, backward=self.look_back)
        bv = look_around(bv, backward=self.look_back)
        bq_k = look_around(b_n, backward=self.look_back)
        return b_n, bq, bk, bv, bq_k

    def _apply_masks(
        self,
        b: int,
        energy: Tensor,
        b_n: Tensor,
        bq_k: Tensor,
        mask: Tensor,
        num_windows: int,
    ) -> Tensor:
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
            backward=self.look_back,
            window_size=self.window_size,
            num_windows=num_windows,
            mask_value=mask_value,
        )
        return energy

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Computes windowed attention."""
        b, n, _ = x.shape

        if not n % self.window_size:
            RuntimeError(
                f"Sequence length {n} must be divisable with window size {self.window_size}"
            )

        num_windows = n // self.window_size

        q, k, v = self._to_embeddings(x)
        d = q.shape[-1]

        # Compute buckets
        b_n, bq, bk, bv, bq_k = self._create_buckets(q, k, v, n, b, num_windows)

        # Compute the attention.
        energy = einsum("b h i d, b h j d -> b h i j", bq, bk) * self.scale
        energy = self._apply_masks(b, energy, b_n, bq_k, mask, num_windows)

        attn = F.softmax(energy, dim=-1)
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, bv)
        out = out.reshape(-1, n, d)

        out = out.reshape(b, self.num_heads, n, -1).transpose(1, 2).reshape(b, n, -1)
        out = self.fc(out)
        return out, attn


def _apply_rotary_emb(q: Tensor, k: Tensor, freqs: Tensor) -> Tuple[Tensor, Tensor]:
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k


def _reshape_dim(t: Tensor, dim: int, split_dims: Tuple[int, int]) -> Tensor:
    shape = list(t.shape)
    dims = len(t.shape)
    dim = (dim + dims) % dims
    shape[dim : dim + 1] = split_dims
    return t.reshape(shape)


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


def look_around(x: Tensor, backward: int, pad_value: int = -1, dim: int = 2) -> Tensor:
    """Apply windowing."""
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
    """Applies input mask to energy tensor."""
    h = b // mask.shape[0]
    mask = mask.reshape(-1, num_windows, window_size)
    mq = mk = mask
    mk = look_around(mk, pad_value=False, backward=backward)
    mask = mq[:, :, :, None] * mk[:, :, None, :]
    mask = merge_dims(0, 1, expand_dim(mask, 1, h))
    energy.masked_fill_(~mask, mask_value)
    del mask
    return energy

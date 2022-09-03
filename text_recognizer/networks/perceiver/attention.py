"""Attention module."""
from typing import Optional

from einops import rearrange, repeat
import torch
from torch import einsum, nn, Tensor
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
    ) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(
        self, x: Tensor, context: Optional[Tensor] = None, mask=Optional[Tensor]
    ) -> Tensor:
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, v, k = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        # if mask is not None:
        #     mask = rearrange(mask, "b ... -> b (...)")
        #     max_neg_val = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, "b j -> (b h) () j", h=h)
        #     sim.masked_fill_(~mask, max_neg_val)

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)

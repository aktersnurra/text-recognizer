"""Implements the attention module for the transformer."""
from typing import Optional
from text_recognizer.network.transformer.norm import RMSNorm
from text_recognizer.network.transformer.attend import Attend

import torch
from einops import rearrange
from torch import Tensor, nn


class Attention(nn.Module):
    """Standard attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        causal: bool = False,
        dim_head: int = 64,
        dropout_rate: float = 0.0,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        # self.q_norm = RMSNorm(heads, dim_head)
        # self.k_norm = RMSNorm(heads, dim_head)
        self.attend = Attend(use_flash)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.scale = dim**-0.5
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes the attention."""
        x = self.norm(x)
        q = self.to_q(x)
        k = self.to_k(x if context is None else context)
        v = self.to_v(x if context is None else context)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )
        out = self.attend(q, k, v, self.causal, mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out

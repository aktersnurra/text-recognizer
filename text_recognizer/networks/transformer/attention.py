"""Implementes the attention module for the transformer."""
from typing import Optional, Tuple

from einops.layers.torch import Rearrange
import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from text_recognizer.networks.transformer.rotary_embedding import apply_rotary_pos_emb


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int = 64,
        dropout_rate: float = 0.0,
        causal: bool = False,
    ) -> None:
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.causal = causal
        inner_dim = dim * dim_head

        # Attnetion
        self.qkv_fn = nn.Sequential(
            nn.Linear(dim, 3 * inner_dim, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads),
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.attn_fn = F.softmax

        # Feedforward
        self.proj = nn.Linear(inner_dim, dim)

    @staticmethod
    def _apply_rotary_emb(
        q: Tensor, k: Tensor, rotary_pos_emb: Tensor
    ) -> Tuple[Tensor, Tensor]:
        l = rotary_pos_emb.shape[-1]
        (ql, qr), (kl, kr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k))
        ql, kl = apply_rotary_pos_emb(ql, kl, rotary_pos_emb)
        q = torch.cat((ql, qr), dim=-1)
        k = torch.cat((kl, kr), dim=-1)
        return q, k

    def _cross_attention(self) -> Tensor:
        pass

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor],
        mask: Optional[Tensor],
        context_mask: Optional[Tensor],
        rotary_pos_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        b, n, _, device = x.shape, x.device
        q, k, v = self.qkv_fn(x)
        q, k = (
            self._apply_rotary_emb(q, k, rotary_pos_emb)
            if rotary_pos_emb is not None
            else q,
            k,
        )

        input_mask = None
        if any(x is not None for x in (mask, context_mask)):
            q_mask = (
                mask
                if mask is not None
                else lambda: torch.ones((b, n), device=device).bool()
            )
            pass

        # Compute the attention
        energy = (q @ k.transpose(-2, -1)) * self.scale

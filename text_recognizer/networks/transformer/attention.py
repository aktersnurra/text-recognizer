"""Implementes the attention module for the transformer."""
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn

from text_recognizer.networks.transformer.embeddings.rotary import (
    RotaryEmbedding,
)


class Attention(nn.Module):
    """Standard attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        causal: bool = False,
        dim_head: int = 64,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.scale = self.dim**-0.5
        self.num_heads = num_heads
        self.dim_head = dim_head

        self.causal = causal
        self.dropout_rate = dropout_rate

        # Single key/value head
        k_dim = dim_head
        v_dim = dim_head

        out_dim = self.num_heads * self.dim_head

        self.to_q = nn.Linear(self.dim, out_dim, bias=False)
        self.to_k = nn.Linear(self.dim, k_dim, bias=False)
        self.to_v = nn.Linear(self.dim, v_dim, bias=False)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Feedforward
        self.fc = nn.Linear(out_dim, self.dim)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        rotary_embedding: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
        """Computes the attention."""
        b, device = x.shape[0], x.device

        q = self.to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = self.to_k(context) if context is not None else self.to_k(x)
        v = self.to_v(context) if context is not None else self.to_v(x)

        if rotary_embedding is not None:
            q, k, v = map(lambda t: rotary_embedding.rotate(t), (q, k, v))

        energy = einsum("b h i d, b j d -> b h i j", q, k) * self.scale
        mask_value = -torch.finfo(energy.dtype).max
        energy = apply_input_mask(b, k, energy, mask, mask_value, device)
        if self.causal:
            energy = apply_causal_mask(energy, mask, mask_value, device)

        attn = F.softmax(energy, dim=-1)
        attn = self.dropout(attn)
        out = einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.fc(out)
        return out


def apply_input_mask(
    b: int,
    k: Tensor,
    energy: Tensor,
    mask: Optional[Tensor],
    mask_value: Tensor,
    device: str,
) -> Tensor:
    """Applies an input mask."""
    if mask is not None:
        k_mask = torch.ones((b, k.shape[-2]), device=device).bool()
        q_mask = rearrange(mask, "b i -> b () i ()")
        k_mask = rearrange(k_mask, "b j -> b () () j")
        input_mask = q_mask * k_mask

        energy = energy.masked_fill_(~input_mask, mask_value)
    return energy


def apply_causal_mask(
    energy: Tensor, mask: Tensor, mask_value: Tensor, device: str
) -> Tensor:
    """Applies a causal mask to the energy tensor."""
    i, j = energy.shape[-2:]
    r = torch.arange(i, device=device)
    mask = rearrange(r, "i -> () () i ()") < rearrange(r, "j -> () () () j")
    mask = F.pad(mask, (j - i, 0), value=False)
    energy.masked_fill_(mask, mask_value)
    return energy

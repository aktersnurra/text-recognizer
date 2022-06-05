"""Implementes the attention module for the transformer."""
from typing import Optional, Tuple

from einops import rearrange
import torch
from torch import einsum
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from text_recognizer.networks.transformer.embeddings.rotary import (
    RotaryEmbedding,
    rotate_half,
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
        rotary_embedding: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.causal = causal
        self.dim_head = dim_head
        self.dropout_rate = dropout_rate
        self.rotary_embedding = rotary_embedding

        self.scale = self.dim ** -0.5
        inner_dim = self.num_heads * self.dim_head

        self.to_q = nn.Linear(self.dim, inner_dim, bias=False)
        self.to_k = nn.Linear(self.dim, inner_dim, bias=False)
        self.to_v = nn.Linear(self.dim, inner_dim, bias=False)

        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Feedforward
        self.fc = nn.Linear(inner_dim, self.dim)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes the attention."""
        b, n, _, device = *x.shape, x.device

        q = self.to_q(x)
        k = self.to_k(context) if context is not None else self.to_k(x)
        v = self.to_v(context) if context is not None else self.to_v(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )

        if self.rotary_embedding is not None:
            embedding = self.rotary_embedding(q)
            q, k, v = _apply_rotary_emb(q, k, v, embedding[None, ...])

        energy = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = -torch.finfo(energy.dtype).max
        energy = apply_input_mask(
            b, n, k, energy, input_mask, context, context_mask, mask_value, device
        )
        if self.causal:
            energy = apply_causal_mask(energy, input_mask, mask_value, device)

        attn = F.softmax(energy, dim=-1)
        attn = self.dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.fc(out)
        return out


def _apply_rotary_emb(
    q: Tensor, k: Tensor, v: Tensor, freqs: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    q, k, v = map(
        lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k, v)
    )
    return q, k, v


def apply_input_mask(
    b: int,
    n: int,
    k: Tensor,
    energy: Tensor,
    input_mask: Optional[Tensor],
    context: Optional[Tensor],
    context_mask: Optional[Tensor],
    mask_value: Tensor,
    device: str,
) -> Tensor:
    """Applies an input mask."""
    if any(x is not None for x in (input_mask, context_mask)):
        q_mask = (
            input_mask
            if input_mask is not None
            else torch.ones((b, n), device=device).bool()
        )
        k_mask = q_mask if context is None else context_mask
        k_mask = (
            torch.ones((b, k.shape[-2]), device=device).bool()
            if k_mask is None
            else k_mask
        )
        q_mask = rearrange(q_mask, "b i -> b () i ()")
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

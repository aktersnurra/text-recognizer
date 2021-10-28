"""Implementes the attention module for the transformer."""
from typing import Optional, Tuple

import attr
from einops import rearrange
import torch
from torch import einsum
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from text_recognizer.networks.transformer.embeddings.rotary import apply_rotary_pos_emb


@attr.s(eq=False)
class Attention(nn.Module):
    """Standard attention."""

    def __attrs_pre_init__(self) -> None:
        """Pre init constructor."""
        super().__init__()

    dim: int = attr.ib()
    num_heads: int = attr.ib()
    causal: bool = attr.ib(default=False)
    dim_head: int = attr.ib(default=64)
    dropout_rate: float = attr.ib(default=0.0)
    scale: float = attr.ib(init=False)
    dropout: nn.Dropout = attr.ib(init=False)
    fc: nn.Linear = attr.ib(init=False)
    qkv_fn: nn.Sequential = attr.ib(init=False)

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
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Computes the attention."""
        b, n, _, device = *x.shape, x.device

        q = self.query(x)
        k = self.key(context) if context is not None else self.key(x)
        v = self.value(context) if context is not None else self.value(x)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )
        q, k, v = (
            apply_rotary_emb(q, k, v, rotary_pos_emb)
            if rotary_pos_emb is not None and context is None
            else (q, k, v,)
        )

        energy = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = -torch.finfo(energy.dtype).max
        energy = apply_input_mask(
            b, n, k, energy, mask, context, context_mask, mask_value, device
        )
        if self.causal:
            energy = apply_causal_mask(energy, mask, mask_value, device)

        attn = F.softmax(energy, dim=-1)
        attn = self.dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.fc(out)
        return out, attn


def apply_rotary_emb(
    q: Tensor, k: Tensor, v: Tensor, rotary_pos_emb: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Applies rotary embedding."""
    emb_len = rotary_pos_emb.shape[-1]
    (ql, qr), (kl, kr), (vl, vr) = map(
        lambda t: (t[..., :emb_len], t[..., emb_len:]), (q, k, v)
    )
    ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))
    q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))
    return q, k, v


def apply_input_mask(
    b: int,
    n: int,
    k: Tensor,
    energy: Tensor,
    mask: Optional[Tensor],
    context: Optional[Tensor],
    context_mask: Optional[Tensor],
    mask_value: Tensor,
    device: str,
) -> Tensor:
    """Applies an input mask."""
    if any(x is not None for x in (mask, context_mask)):
        q_mask = mask if mask is not None else torch.ones((b, n), device=device).bool()
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
        del input_mask
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
    del mask
    return energy

"""Implements the attention module for the transformer."""
from typing import Optional

from einops import rearrange
from text_recognizer.network.transformer.swiglu import SwiGLU
import torch
from torch import Tensor, nn

from .attend import Attend
from .embedding.rotary import RotaryEmbedding, apply_rotary_pos_emb


class Attention(nn.Module):
    """Standard attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        causal: bool = False,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout_rate: float = 0.0,
        use_flash: bool = True,
        norm_context: bool = False,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.scale = dim**-0.5
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim) if norm_context else nn.Identity()
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, 2 * inner_dim, bias=False)

        self.attend = Attend(use_flash)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.rotary_emb = rotary_emb
        self.pos_emb = None
        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * ff_inner_dim), SwiGLU(), nn.Linear(ff_inner_dim, dim)
        )

    def get_rotary_embedding(self, n: int, device: torch.device) -> Tensor:
        assert self.rotary_emb is not None, "No rotary embedding"
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n].to(device)
        self.pos_emb = self.rotary_emb(n, device=device)
        return self.pos_emb

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes the attention."""
        x = self.norm(x)
        q = self.to_q(x)
        k, v = self.to_kv(x if context is None else self.context_norm(context)).chunk(
            2, dim=-1
        )

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        if self.rotary_emb is not None:
            pos_emb = self.get_rotary_embedding(x.shape[1], x.device)
            q, k = map(lambda t: apply_rotary_pos_emb(pos_emb, t), (q, k))

        out = self.attend(q, k, v, self.causal, mask)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out + self.ff(x)

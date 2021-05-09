"""Transformer wrapper."""
from typing import Optional, Type

from torch import nn, Tensor

from .layers import AttentionLayers
from text_recognizer.networks.transformer.positional_encodings import (
    AbsolutePositionalEmbedding,
)


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int,
        attn_layers: Type[AttentionLayers],
        emb_dim: Optional[int] = None,
        emb_dropout: float = 0.0,
        use_pos_emb: bool = True,
    ) -> None:
        dim = attn_layers.dim
        emb_dim = emb_dim if emb_dim is not None else dim
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.pos_emb = (
            AbsolutePositionalEmbedding(emb_dim, max_seq_len)
            if (use_pos_emb and not self.attn_layers.has_pos_emb)
            else None
        )

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self._init_weights()

        self.logits = nn.Linear(dim, num_tokens)

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor],
        return_embeddings: bool = False,
        **kwargs: Any
    ) -> Tensor:
        b, n, device = *x.shape, x.device
        x += self.token_emb(x)
        if self.pos_emb is not None:
            x += self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)
        x = self.attn_layers(x, mask=mask, **kwargs)
        out = self.logits(x) if not return_embeddings else x
        return x

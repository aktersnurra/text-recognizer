"""Transformer decoder module."""
from typing import Optional
from torch import Tensor, nn

from text_recognizer.network.transformer.attention import Attention
from text_recognizer.network.transformer.ff import FeedForward


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        heads: int,
        dim_head: int,
        depth: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads,
                            True,
                            dim_head,
                            dropout_rate,
                        ),
                        FeedForward(dim, inner_dim, dropout_rate),
                        Attention(
                            dim,
                            heads,
                            False,
                            dim_head,
                            dropout_rate,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        for self_attn, ff, cross_attn in self.layers:
            x = x + self_attn(x, mask=mask)
            x = x + ff(x)
            x = x + cross_attn(x, context=context)
        return self.norm(x)

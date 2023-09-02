"""Transformer encoder module."""
from torch import Tensor, nn

from .attention import Attention
from .ff import FeedForward


class Encoder(nn.Module):
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
                            False,
                            dim_head,
                            dropout_rate,
                        ),
                        FeedForward(dim, inner_dim, dropout_rate),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        for self_attn, ff in self.layers:
            x = x + self_attn(x)
            x = x + ff(x)
        return self.norm(x)

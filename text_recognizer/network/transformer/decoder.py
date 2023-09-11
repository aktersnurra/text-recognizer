"""Transformer decoder module."""
from typing import Optional
from torch import Tensor, nn

from .attention import Attention
from .embedding.rotary import RotaryEmbedding


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: int,
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
                            dim=dim,
                            heads=heads,
                            causal=True,
                            dim_head=dim_head,
                            ff_mult=ff_mult,
                            dropout_rate=dropout_rate,
                            use_flash=True,
                            norm_context=False,
                            rotary_emb=RotaryEmbedding(dim_head),
                        ),
                        Attention(
                            dim=dim,
                            heads=heads,
                            causal=False,
                            dim_head=dim_head,
                            ff_mult=ff_mult,
                            dropout_rate=dropout_rate,
                            use_flash=True,
                            norm_context=False,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def self_attn(self, x: Tensor, mask: Tensor) -> Tensor:
        for self_attn, _ in self.layers:
            x = x + self_attn(x, mask=mask)
        return self.norm(x)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        for self_attn, cross_attn in self.layers:
            x = x + self_attn(x, mask=mask)
            x = x + cross_attn(x, context=context)
        return self.norm(x)

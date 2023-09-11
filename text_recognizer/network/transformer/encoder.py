"""Transformer encoder module."""
from torch import Tensor, nn

from .attention import Attention


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: int,
        depth: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                Attention(
                    dim=dim,
                    heads=heads,
                    causal=False,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout_rate=dropout_rate,
                    use_flash=True,
                    norm_context=False,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        for self_attn in self.layers:
            x = x + self_attn(x)
        return self.norm(x)

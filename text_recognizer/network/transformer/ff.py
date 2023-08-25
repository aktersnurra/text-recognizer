"""Feedforward layer in transformer."""
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ff(x)

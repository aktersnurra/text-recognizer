"""Conformer feedforward block."""
from torch import nn, Tensor


class Feedforward(nn.Module):
    def __init__(
        self, dim: int, expansion_factor: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, expansion_factor * dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

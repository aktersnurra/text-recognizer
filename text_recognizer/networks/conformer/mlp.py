"""Conformer feedforward block."""
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, mult * dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

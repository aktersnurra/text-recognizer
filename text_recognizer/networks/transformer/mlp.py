"""Feedforward layer in transformer.

Stolen from lucidrains:
    https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
"""
from typing import Optional

from torch import nn
from torch import Tensor
import torch.nn.functional as F


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.fc(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        expansion_factor: int = 4,
        glu: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim * expansion_factor
        dim_out = dim_out if dim_out is not None else dim
        in_projection = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.mlp = nn.Sequential(
            in_projection, nn.Dropout(dropout_rate), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

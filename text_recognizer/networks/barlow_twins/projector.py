"""Projector network in Barlow Twins."""

from typing import List
import torch
from torch import nn
from torch import Tensor


class Projector(nn.Module):
    """MLP network."""

    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        self.dims = dims
        self.network = self._build()

    def _build(self) -> nn.Sequential:
        """Builds projector network."""
        layers = [
            nn.Sequential(
                nn.Linear(
                    in_features=self.dims[i], out_features=self.dims[i + 1], bias=False
                ),
                nn.BatchNorm1d(self.dims[i + 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(len(self.dims) - 2)
        ]
        layers.append(
            nn.Linear(in_features=self.dims[-2], out_features=self.dims[-1], bias=False)
        )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Project latent to higher dimesion."""
        return self.network(x)

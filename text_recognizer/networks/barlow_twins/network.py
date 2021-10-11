"""Barlow Twins network."""
from typing import Type

from torch import nn, Tensor
import torch.nn.functional as F


class BarlowTwins(nn.Module):
    def __init__(self, encoder: Type[nn.Module], projector: Type[nn.Module]) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z_e = F.adaptive_avg_pool2d(z, (1, 1)).flatten(start_dim=1)
        z_p = self.projector(z_e)
        return z_p


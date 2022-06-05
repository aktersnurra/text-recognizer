"""Conformer module."""
from copy import deepcopy

from torch import nn, Tensor

from text_recognizer.networks.conformer.block import ConformerBlock


class Conformer(nn.Module):
    def __init__(self, block: ConformerBlock, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([deepcopy(block) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        for fn in self.blocks:
            x = fn(x)
        return x

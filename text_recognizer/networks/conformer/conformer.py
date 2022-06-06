"""Conformer module."""
from copy import deepcopy
from typing import Type

from torch import nn, Tensor

from text_recognizer.networks.conformer.block import ConformerBlock


class Conformer(nn.Module):
    def __init__(
        self,
        subsampler: Type[nn.Module],
        block: ConformerBlock,
        depth: int,
    ) -> None:
        super().__init__()
        self.subsampler = subsampler
        self.blocks = nn.ModuleList([deepcopy(block) for _ in range(depth)])

    def forward(self, x: Tensor) -> Tensor:
        x = self.subsampler(x)
        for fn in self.blocks:
            x = fn(x)
        return x

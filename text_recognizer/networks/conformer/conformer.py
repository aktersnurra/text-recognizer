"""Conformer module."""
from copy import deepcopy
from typing import Type

from torch import nn, Tensor

from text_recognizer.networks.conformer.block import ConformerBlock


class Conformer(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_gru: int,
        num_classes: int,
        subsampler: Type[nn.Module],
        block: ConformerBlock,
        depth: int,
    ) -> None:
        super().__init__()
        self.subsampler = subsampler
        self.blocks = nn.ModuleList([deepcopy(block) for _ in range(depth)])
        self.gru = nn.GRU(
            dim, dim_gru, 1, bidirectional=True, batch_first=True, bias=False
        )
        self.fc = nn.Linear(dim_gru, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.subsampler(x)
        B, T, C = x.shape
        for fn in self.blocks:
            x = fn(x)
        x, _ = self.gru(x)
        x = x.view(B, T, 2, -1).sum(2)
        return self.fc(x)

"""Axial attention for multi-dimensional data.

Stolen from:
    https://github.com/lucidrains/axial-attention/blob/eff2c10c2e76c735a70a6b995b571213adffbbb7/axial_attention/axial_attention.py#L100
"""
from typing import Sequence

import torch
from torch import nn, Tensor


class AxialPositionalEmbedding(nn.Module):
    """Axial positional embedding."""

    def __init__(self, dim: int, shape: Sequence[int], emb_dim_index: int = 1) -> None:
        super().__init__()
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f"param_{i}", parameter)

    def forward(self, x: Tensor) -> Tensor:
        """Applies axial positional embedding."""
        for i in range(self.num_axials):
            x = x + getattr(self, f"param_{i}")
        return x

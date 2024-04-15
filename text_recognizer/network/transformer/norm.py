"""Normalization layers for transformers.

Copied from lucidrains:
    https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, heads: int, dim: int) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Applies normalization."""
        return F.normalize(x, dim=-1) * self.scale * self.gamma

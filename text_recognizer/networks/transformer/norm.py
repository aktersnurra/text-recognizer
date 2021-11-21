"""Normalization layers for transfromers.

Copied from lucidrains:
    https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

"""
from typing import Dict, Type

import torch
from torch import nn
from torch import Tensor


class ScaleNorm(nn.Module):
    """Scaled normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.scale = normalized_shape ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """Applies scale norm."""
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class PreNorm(nn.Module):
    """Applies layer normalization then function."""

    def __init__(self, normalized_shape: int, fn: Type[nn.Module]) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs: Dict) -> Tensor:
        """Applies pre norm."""
        x = self.norm(x)
        return self.fn(x, **kwargs)

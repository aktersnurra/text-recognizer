"""Normalization layers for transfromers.

Copied from lucidrains:
    https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

"""
from typing import Callable, Dict

import torch
from torch import nn
from torch import Tensor


class Rezero(nn.Module):
    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, **kwargs: Dict) -> Tensor:
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) self.g
    

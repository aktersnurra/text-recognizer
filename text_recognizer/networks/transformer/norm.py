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

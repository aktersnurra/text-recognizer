"""Generic residual layer."""
from typing import Callable
from torch import nn, Tensor


class Residual(nn.Module):
    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x

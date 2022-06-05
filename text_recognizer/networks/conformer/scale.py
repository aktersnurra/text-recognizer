"""Scale layer."""
from typing import Dict
from torch import nn, Tensor


class Scale(nn.Module):
    def __init__(self, scale: float, fn: nn.Module) -> None:
        super().__init__()
        self.scale = scale
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(x, **kwargs) * self.scale

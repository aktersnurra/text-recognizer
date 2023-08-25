"""Generic residual layer."""
from typing import Callable

from torch import Tensor, nn


class Residual(nn.Module):
    """Residual layer."""

    def __init__(self, fn: Callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        """Applies residual fn."""
        return self.fn(x) + x

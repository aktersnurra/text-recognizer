"""Residual function."""
from torch import nn, Tensor


class Residual(nn.Module):
    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        """Applies the residual function."""
        return x + residual

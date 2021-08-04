"""Residual block."""
from torch import nn
from torch import Tensor


class Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Mish(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Mish(inplace=True),
            nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual forward pass."""
        return x + self.block(x)

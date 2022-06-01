"""Depthwise 1D convolution."""
from torch import nn, Tensor


class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding="same",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

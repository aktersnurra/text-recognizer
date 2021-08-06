"""Up and down-sample with linear interpolation."""
from torch import nn, Tensor
import torch.nn.functional as F


class Upsample(nn.Module):
    """Upsamples by a factor 2."""

    def forward(self, x: Tensor) -> Tensor:
        """Applies upsampling."""
        return F.interpolate(x, scale_factor=2.0, mode="nearest")


class Downsample(nn.Module):
    """Downsampling by a factor 2."""

    def forward(self, x: Tensor) -> Tensor:
        """Applies downsampling."""
        return F.avg_pool2d(x, kernel_size=2, stride=2)

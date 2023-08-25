"""Convnext downsample module."""
from typing import Tuple

from einops.layers.torch import Rearrange
from torch import Tensor, nn


class Downsample(nn.Module):
    """Downsamples feature maps by patches."""

    def __init__(self, dim: int, dim_out: int, factors: Tuple[int, int]) -> None:
        super().__init__()
        s1, s2 = factors
        self.fn = nn.Sequential(
            Rearrange("b c (h s1) (w s2) -> b (c s1 s2) h w", s1=s1, s2=s2),
            nn.Conv2d(dim * s1 * s2, dim_out, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies patch function."""
        return self.fn(x)

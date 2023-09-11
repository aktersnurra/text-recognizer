"""Convnext downsample module."""
from einops.layers.torch import Rearrange
from torch import Tensor, nn


class Downsample(nn.Module):
    """Downsamples feature maps by patches."""

    def __init__(self, dim: int, dim_out: int) -> None:
        super().__init__()
        self.fn = nn.Sequential(
            Rearrange("b c (h s1) (w s2) -> b (c s1 s2) h w", s1=2, s2=2),
            nn.Conv2d(dim * 4, dim_out, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies patch function."""
        return self.fn(x)

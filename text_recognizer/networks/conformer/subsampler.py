"""Simple convolutional network."""
from typing import Tuple

from einops import rearrange
from torch import nn, Tensor

from text_recognizer.networks.transformer import AxialPositionalEmbedding


class Subsampler(nn.Module):
    def __init__(
        self,
        channels: int,
        dim: int,
        depth: int,
        height: int,
        pixel_pos_embedding: AxialPositionalEmbedding,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pixel_pos_embedding = pixel_pos_embedding
        self.subsampler, self.projector = self._build(
            channels, height, dim, depth, dropout
        )

    def _build(
        self, channels: int, height: int, dim: int, depth: int, dropout: float
    ) -> Tuple[nn.Sequential, nn.Sequential]:
        subsampler = []
        for i in range(depth):
            subsampler.append(
                nn.Conv2d(
                    in_channels=1 if i == 0 else channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=2,
                )
            )
            subsampler.append(nn.Mish(inplace=True))
        projector = nn.Sequential(
            nn.Linear(channels * height, dim), nn.Dropout(dropout)
        )
        return nn.Sequential(*subsampler), projector

    def forward(self, x: Tensor) -> Tensor:
        x = self.subsampler(x)
        x = self.pixel_pos_embedding(x)
        x = rearrange(x, "b c h w -> b w (c h)")
        x = self.projector(x)
        return x

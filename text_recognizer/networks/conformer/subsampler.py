"""Simple convolutional network."""
from typing import Tuple

from torch import nn, Tensor

from text_recognizer.networks.transformer import (
    AxialPositionalEmbedding,
)


class Subsampler(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        pixel_pos_embedding: AxialPositionalEmbedding,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pixel_pos_embedding = pixel_pos_embedding
        self.subsampler, self.projector = self._build(channels, depth, dropout)

    def _build(
        self, channels: int, depth: int, dropout: float
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
            nn.Flatten(start_dim=2), nn.Linear(channels, channels), nn.Dropout(dropout)
        )
        return nn.Sequential(*subsampler), projector

    def forward(self, x: Tensor) -> Tensor:
        x = self.subsampler(x)
        x = self.pixel_pos_embedding(x)
        x = self.projector(x)
        return x.permute(0, 2, 1)

"""Simple convolutional network."""
import torch
from torch import nn, Tensor


class CNN(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.layers = self._build(channels, depth)

    def _build(self, channels: int, depth: int) -> nn.Sequential:
        layers = []
        for i in range(depth):
            layers.append(
                nn.Conv2d(
                    in_channels=1 if i == 0 else channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=2,
                )
            )
            layers.append(nn.Mish(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

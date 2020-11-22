"""UNet for segmentation."""
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function


class ConvBlock(nn.Module):
    """Basic UNet convolutional block."""

    def __init__(self, channels: List[int], activation: str) -> None:
        super().__init__()
        self.channels = channels
        self.activation = activation_function(activation)
        self.block = self._configure_block()

    def _configure_block(self) -> nn.Sequential:
        block = []
        for i in range(len(self.channels) - 1):
            block += [
                nn.Conv2d(
                    self.channels[i], self.channels[i + 1], kernel_size=3, padding=1
                ),
                nn.BatchNorm2d(self.channels[i + 1]),
                self.activation,
            ]

        return nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolutional block."""
        return self.block(x)


class DownSamplingBlock(nn.Module):
    """Basic down sampling block."""

    def __init__(
        self,
        channels: List[int],
        activation: str,
        pooling_kernel: Union[int, bool] = 2,
    ) -> None:
        super().__init__()
        self.conv_block = ConvBlock(channels, activation)
        self.down_sampling = nn.MaxPool2d(pooling_kernel) if pooling_kernel else None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the convolutional block output and a down sampled tensor."""
        x = self.conv_block(x)
        if self.down_sampling is not None:
            x_down = self.down_sampling(x)
        else:
            x_down = None
        return x_down, x


class UpSamplingBlock(nn.Module):
    """The upsampling block of the UNet."""

    def __init__(
        self, channels: List[int], activation: str, scale_factor: int = 2
    ) -> None:
        super().__init__()
        self.conv_block = ConvBlock(channels, activation)
        self.up_sampling = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

    def forward(self, x: Tensor, x_skip: Optional[Tensor] = None) -> Tensor:
        """Apply the up sampling and convolutional block."""
        x = self.up_sampling(x)
        if x_skip is not None:
            x = torch.cat((x, x_skip), dim=1)
        return self.conv_block(x)


class UNet(nn.Module):
    """UNet architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 3,
        depth: int = 4,
        out_channels: int = 3,
        activation: str = "relu",
        pooling_kernel: int = 2,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        channels = [1] + [base_channels * 2 ** i for i in range(depth)]
        self.encoder_blocks = self._configure_down_sampling_blocks(
            channels, activation, pooling_kernel
        )
        self.decoder_blocks = self._configure_up_sampling_blocks(
            channels, activation, scale_factor
        )

        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _configure_down_sampling_blocks(
        self, channels: List[int], activation: str, pooling_kernel: int
    ) -> nn.ModuleList:
        blocks = nn.ModuleList([])
        for i in range(len(channels) - 1):
            pooling_kernel = pooling_kernel if i < self.depth - 1 else False
            blocks += [
                DownSamplingBlock(
                    [channels[i], channels[i + 1], channels[i + 1]],
                    activation,
                    pooling_kernel,
                )
            ]

        return blocks

    def _configure_up_sampling_blocks(
        self,
        channels: List[int],
        activation: str,
        scale_factor: int,
    ) -> nn.ModuleList:
        channels.reverse()
        return nn.ModuleList(
            [
                UpSamplingBlock(
                    [channels[i] + channels[i + 1], channels[i + 1], channels[i + 1]],
                    activation,
                    scale_factor,
                )
                for i in range(len(channels) - 2)
            ]
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        x_skips = []
        for block in self.encoder_blocks:
            x, x_skip = block(x)
            if x_skip is not None:
                x_skips.append(x_skip)
        return x, x_skips

    def decode(self, x: Tensor, x_skips: List[Tensor]) -> Tensor:
        x = x_skips[-1]
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, x_skips[-(i + 2)])
        return x

    def forward(self, x: Tensor) -> Tensor:
        x, x_skips = self.encode(x)
        x = self.decode(x, x_skips)
        return self.head(x)

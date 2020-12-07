"""UNet for segmentation."""
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function


class _ConvBlock(nn.Module):
    """Modified UNet convolutional block with dilation."""

    def __init__(
        self,
        channels: List[int],
        activation: str,
        num_groups: int,
        dropout_rate: float = 0.1,
        kernel_size: int = 3,
        dilation: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.num_groups = num_groups
        self.activation = activation_function(activation)
        self.block = self._configure_block()
        self.residual_conv = nn.Sequential(
            nn.Conv2d(
                self.channels[0], self.channels[-1], kernel_size=3, stride=1, padding=1
            ),
            self.activation,
        )

    def _configure_block(self) -> nn.Sequential:
        block = []
        for i in range(len(self.channels) - 1):
            block += [
                nn.Dropout(p=self.dropout_rate),
                nn.GroupNorm(self.num_groups, self.channels[i]),
                self.activation,
                nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=1,
                    dilation=self.dilation,
                ),
            ]

        return nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolutional block."""
        residual = self.residual_conv(x)
        return self.block(x) + residual


class _DownSamplingBlock(nn.Module):
    """Basic down sampling block."""

    def __init__(
        self,
        channels: List[int],
        activation: str,
        num_groups: int,
        pooling_kernel: Union[int, bool] = 2,
        dropout_rate: float = 0.1,
        kernel_size: int = 3,
        dilation: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv_block = _ConvBlock(
            channels,
            activation,
            num_groups,
            dropout_rate,
            kernel_size,
            dilation,
            padding,
        )
        self.down_sampling = nn.MaxPool2d(pooling_kernel) if pooling_kernel else None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the convolutional block output and a down sampled tensor."""
        x = self.conv_block(x)
        x_down = self.down_sampling(x) if self.down_sampling is not None else x

        return x_down, x


class _UpSamplingBlock(nn.Module):
    """The upsampling block of the UNet."""

    def __init__(
        self,
        channels: List[int],
        activation: str,
        num_groups: int,
        scale_factor: int = 2,
        dropout_rate: float = 0.1,
        kernel_size: int = 3,
        dilation: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv_block = _ConvBlock(
            channels,
            activation,
            num_groups,
            dropout_rate,
            kernel_size,
            dilation,
            padding,
        )
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
        activation: str = "relu",
        num_groups: int = 8,
        dropout_rate: float = 0.1,
        pooling_kernel: int = 2,
        scale_factor: int = 2,
        kernel_size: Optional[List[int]] = None,
        dilation: Optional[List[int]] = None,
        padding: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.num_groups = num_groups

        if kernel_size is not None and dilation is not None and padding is not None:
            if (
                len(kernel_size) != depth
                and len(dilation) != depth
                and len(padding) != depth
            ):
                raise RuntimeError(
                    "Length of convolutional parameters does not match the depth."
                )
            self.kernel_size = kernel_size
            self.padding = padding
            self.dilation = dilation

        else:
            self.kernel_size = [3] * depth
            self.padding = [1] * depth
            self.dilation = [1] * depth

        self.dropout_rate = dropout_rate
        self.conv = nn.Conv2d(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1
        )

        channels = [base_channels] + [base_channels * 2 ** i for i in range(depth)]
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
            dropout_rate = self.dropout_rate if i < 0 else 0
            blocks += [
                _DownSamplingBlock(
                    [channels[i], channels[i + 1], channels[i + 1]],
                    activation,
                    self.num_groups,
                    pooling_kernel,
                    dropout_rate,
                    self.kernel_size[i],
                    self.dilation[i],
                    self.padding[i],
                )
            ]

        return blocks

    def _configure_up_sampling_blocks(
        self, channels: List[int], activation: str, scale_factor: int,
    ) -> nn.ModuleList:
        channels.reverse()
        self.kernel_size.reverse()
        self.dilation.reverse()
        self.padding.reverse()
        return nn.ModuleList(
            [
                _UpSamplingBlock(
                    [channels[i] + channels[i + 1], channels[i + 1], channels[i + 1]],
                    activation,
                    self.num_groups,
                    scale_factor,
                    self.dropout_rate,
                    self.kernel_size[i],
                    self.dilation[i],
                    self.padding[i],
                )
                for i in range(len(channels) - 2)
            ]
        )

    def _encode(self, x: Tensor) -> List[Tensor]:
        x_skips = []
        for block in self.encoder_blocks:
            x, x_skip = block(x)
            x_skips.append(x_skip)
        return x_skips

    def _decode(self, x_skips: List[Tensor]) -> Tensor:
        x = x_skips[-1]
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, x_skips[-(i + 2)])
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with the UNet model."""
        if len(x.shape) < 4:
            x = x[(None,) * (4 - len(x.shape))]
        x = self.conv(x)
        x_skips = self._encode(x)
        x = self._decode(x_skips)
        return self.head(x)

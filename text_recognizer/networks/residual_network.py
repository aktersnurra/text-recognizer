"""Residual CNN."""
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union

from einops.layers.torch import Rearrange, Reduce
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function


class Conv2dAuto(nn.Conv2d):
    """Convolution with auto padding based on kernel size."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


def conv_bn(in_channels: int, out_channels: int, *args, **kwargs) -> nn.Sequential:
    """3x3 convolution with batch norm."""
    conv3x3 = partial(
        Conv2dAuto,
        kernel_size=3,
        bias=False,
    )
    return nn.Sequential(
        conv3x3(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
    )


class IdentityBlock(nn.Module):
    """Residual with identity block."""

    def __init__(
        self, in_channels: int, out_channels: int, activation: str = "relu"
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blocks = nn.Identity()
        self.activation_fn = activation_function(activation)
        self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        residual = x
        if self.apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activation_fn(x)
        return x

    @property
    def apply_shortcut(self) -> bool:
        """Check if shortcut should be applied."""
        return self.in_channels != self.out_channels


class ResidualBlock(IdentityBlock):
    """Residual with nonlinear shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 1,
        downsampling: int = 1,
        *args,
        **kwargs
    ) -> None:
        """Short summary.

        Args:
            in_channels (int): Number of in channels.
            out_channels (int): umber of out channels.
            expansion (int): Expansion factor of the out channels. Defaults to 1.
            downsampling (int): Downsampling factor used in stride. Defaults to 1.
            *args (type): Extra arguments.
            **kwargs (type): Extra key value arguments.

        """
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion = expansion
        self.downsampling = downsampling

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.expanded_channels,
                    kernel_size=1,
                    stride=self.downsampling,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expanded_channels),
            )
            if self.apply_shortcut
            else None
        )

    @property
    def expanded_channels(self) -> int:
        """Computes the expanded output channels."""
        return self.out_channels * self.expansion

    @property
    def apply_shortcut(self) -> bool:
        """Check if shortcut should be applied."""
        return self.in_channels != self.expanded_channels


class BasicBlock(ResidualBlock):
    """Basic ResNet block."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                bias=False,
                stride=self.downsampling,
            ),
            self.activation_fn,
            conv_bn(
                in_channels=self.out_channels,
                out_channels=self.expanded_channels,
                bias=False,
            ),
        )


class BottleNeckBlock(ResidualBlock):
    """Bottleneck block to increase depth while minimizing parameter size."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
            ),
            self.activation_fn,
            conv_bn(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=self.downsampling,
            ),
            self.activation_fn,
            conv_bn(
                in_channels=self.out_channels,
                out_channels=self.expanded_channels,
                kernel_size=1,
            ),
        )


class ResidualLayer(nn.Module):
    """ResNet layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: BasicBlock = BasicBlock,
        num_blocks: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(
                in_channels, out_channels, *args, **kwargs, downsampling=downsampling
            ),
            *[
                block(
                    out_channels * block.expansion,
                    out_channels,
                    downsampling=1,
                    *args,
                    **kwargs
                )
                for _ in range(num_blocks - 1)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.blocks(x)
        return x


class ResidualNetworkEncoder(nn.Module):
    """Encoder network."""

    def __init__(
        self,
        in_channels: int = 1,
        block_sizes: Union[int, List[int]] = (32, 64),
        depths: Union[int, List[int]] = (2, 2),
        activation: str = "relu",
        block: Type[nn.Module] = BasicBlock,
        levels: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.block_sizes = (
            block_sizes if isinstance(block_sizes, list) else [block_sizes] * levels
        )
        self.depths = depths if isinstance(depths, list) else [depths] * levels
        self.activation = activation
        self.gate = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.block_sizes[0],
                kernel_size=7,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.block_sizes[0]),
            activation_function(self.activation),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        self.blocks = self._configure_blocks(block)

    def _configure_blocks(
        self, block: Type[nn.Module], *args, **kwargs
    ) -> nn.Sequential:
        channels = [self.block_sizes[0]] + list(
            zip(self.block_sizes, self.block_sizes[1:])
        )
        blocks = [
            ResidualLayer(
                in_channels=channels[0],
                out_channels=channels[0],
                num_blocks=self.depths[0],
                block=block,
                activation=self.activation,
                *args,
                **kwargs
            )
        ]
        blocks += [
            ResidualLayer(
                in_channels=in_channels * block.expansion,
                out_channels=out_channels,
                num_blocks=num_blocks,
                block=block,
                activation=self.activation,
                *args,
                **kwargs
            )
            for (in_channels, out_channels), num_blocks in zip(
                channels[1:], self.depths[1:]
            )
        ]

        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # If batch dimenstion is missing, it needs to be added.
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.gate(x)
        x = self.blocks(x)
        return x


class ResidualNetworkDecoder(nn.Module):
    """Classification head."""

    def __init__(self, in_features: int, num_classes: int = 80) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.decoder(x)


class ResidualNetwork(nn.Module):
    """Full residual network."""

    def __init__(self, in_channels: int, num_classes: int, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = ResidualNetworkEncoder(in_channels, *args, **kwargs)
        self.decoder = ResidualNetworkDecoder(
            in_features=self.encoder.blocks[-1].blocks[-1].expanded_channels,
            num_classes=num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

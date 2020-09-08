"""Wide Residual CNN."""
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union

from einops.layers.torch import Rearrange, Reduce
import numpy as np
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.misc import activation_function


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Helper function for a 3x3 2d convolution."""
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv_init(module: Type[nn.Module]) -> None:
    """Initializes the weights for convolution and batchnorms."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(module.weight, gain=np.sqrt(2))
        nn.init.constant(module.bias, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.constant(module.weight, 1)
        nn.init.constant(module.bias, 0)


class WideBlock(nn.Module):
    """Block used in WideResNet."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        dropout_rate: float,
        stride: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.activation = activation_function(activation)

        # Build blocks.
        self.blocks = nn.Sequential(
            nn.BatchNorm2d(self.in_planes),
            self.activation,
            conv3x3(in_planes=self.in_planes, out_planes=self.out_planes),
            nn.Dropout(p=self.dropout_rate),
            nn.BatchNorm2d(self.out_planes),
            self.activation,
            conv3x3(
                in_planes=self.out_planes,
                out_planes=self.out_planes,
                stride=self.stride,
            ),
        )

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes,
                    kernel_size=1,
                    stride=self.stride,
                    bias=False,
                ),
            )
            if self._apply_shortcut
            else None
        )

    @property
    def _apply_shortcut(self) -> bool:
        """If shortcut should be applied or not."""
        return self.stride != 1 or self.in_planes != self.out_planes

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        residual = x
        if self._apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x


class WideResidualNetwork(nn.Module):
    """WideResNet for character predictions.

    Can be used for classification or encoding of images to a latent vector.

    """

    def __init__(
        self,
        in_channels: int = 1,
        in_planes: int = 16,
        num_classes: int = 80,
        depth: int = 16,
        width_factor: int = 10,
        dropout_rate: float = 0.0,
        num_layers: int = 3,
        block: Type[nn.Module] = WideBlock,
        activation: str = "relu",
        use_decoder: bool = True,
    ) -> None:
        """The initialization of the WideResNet.

        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            in_planes (int): Number of channels to use in the first output kernel. Defaults to 16.
            num_classes (int): Number of classes. Defaults to 80.
            depth (int): Set the number of blocks to use. Defaults to 16.
            width_factor (int): Factor for scaling the number of channels in the network. Defaults to 10.
            dropout_rate (float): The dropout rate. Defaults to 0.0.
            num_layers (int): Number of layers of blocks. Defaults to 3.
            block (Type[nn.Module]): The default block is WideBlock. Defaults to WideBlock.
            activation (str): Name of the activation to use. Defaults to "relu".
            use_decoder (bool): If True, the network output character predictions, if False, the network outputs a
                latent vector. Defaults to True.

        Raises:
            RuntimeError: If the depth is not of the size `6n+4`.

        """

        super().__init__()
        if (depth - 4) % 6 != 0:
            raise RuntimeError("Wide-resnet depth should be 6n+4")
        self.in_channels = in_channels
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_blocks = (depth - 4) // 6
        self.width_factor = width_factor
        self.num_layers = num_layers
        self.block = block
        self.dropout_rate = dropout_rate
        self.activation = activation_function(activation)

        self.num_stages = [self.in_planes] + [
            self.in_planes * 2 ** n * self.width_factor for n in range(self.num_layers)
        ]
        self.num_stages = list(zip(self.num_stages, self.num_stages[1:]))
        self.strides = [1] + [2] * (self.num_layers - 1)

        self.encoder = nn.Sequential(
            conv3x3(in_planes=self.in_channels, out_planes=self.in_planes),
            *[
                self._configure_wide_layer(
                    in_planes=in_planes,
                    out_planes=out_planes,
                    stride=stride,
                    activation=activation,
                )
                for (in_planes, out_planes), stride in zip(
                    self.num_stages, self.strides
                )
            ],
        )

        self.decoder = (
            nn.Sequential(
                nn.BatchNorm2d(self.num_stages[-1][-1], momentum=0.8),
                self.activation,
                Reduce("b c h w -> b c", "mean"),
                nn.Linear(
                    in_features=self.num_stages[-1][-1], out_features=self.num_classes
                ),
            )
            if use_decoder
            else None
        )

        self.apply(conv_init)

    def _configure_wide_layer(
        self, in_planes: int, out_planes: int, stride: int, activation: str
    ) -> List:
        strides = [stride] + [1] * (self.num_blocks - 1)
        planes = [out_planes] * len(strides)
        planes = [(in_planes, out_planes)] + list(zip(planes, planes[1:]))
        return nn.Sequential(
            *[
                self.block(
                    in_planes=in_planes,
                    out_planes=out_planes,
                    dropout_rate=self.dropout_rate,
                    stride=stride,
                    activation=activation,
                )
                for (in_planes, out_planes), stride in zip(planes, strides)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Feedforward pass."""
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.encoder(x)
        if self.decoder is not None:
            x = self.decoder(x)
        return x

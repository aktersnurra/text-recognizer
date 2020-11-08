"""Defines a Densely Connected Convolutional Networks in PyTorch.

Sources:
https://arxiv.org/abs/1608.06993
https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

"""
from typing import List, Optional, Union

from einops.layers.torch import Rearrange
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function


class _DenseLayer(nn.Module):
    """A dense layer with pre-batch norm -> activation function -> Conv-layer x 2."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_rate: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        activation_fn = activation_function(activation)
        self.dense_layer = [
            nn.BatchNorm2d(in_channels),
            activation_fn,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(bn_size * growth_rate),
            activation_fn,
            nn.Conv2d(
                in_channels=bn_size * growth_rate,
                out_channels=growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        ]
        if dropout_rate:
            self.dense_layer.append(nn.Dropout(p=dropout_rate))

        self.dense_layer = nn.Sequential(*self.dense_layer)

    def forward(self, x: Union[Tensor, List[Tensor]]) -> Tensor:
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.dense_layer(x)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_rate: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.dense_block = self._build_dense_blocks(
            num_layers, in_channels, bn_size, growth_rate, dropout_rate, activation,
        )

    def _build_dense_blocks(
        self,
        num_layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_rate: float,
        activation: str = "relu",
    ) -> nn.ModuleList:
        dense_block = []
        for i in range(num_layers):
            dense_block.append(
                _DenseLayer(
                    in_channels=in_channels + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    dropout_rate=dropout_rate,
                    activation=activation,
                )
            )
        return nn.ModuleList(dense_block)

    def forward(self, x: Tensor) -> Tensor:
        feature_maps = [x]
        for layer in self.dense_block:
            x = layer(feature_maps)
            feature_maps.append(x)
        return torch.cat(feature_maps, 1)


class _Transition(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, activation: str = "relu",
    ) -> None:
        super().__init__()
        activation_fn = activation_function(activation)
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation_fn,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transition(x)


class DenseNet(nn.Module):
    """Implementation of Densenet, a network archtecture that concats previous layers for maximum infomation flow."""

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: List[int] = (6, 12, 24, 16),
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 80,
        bn_size: int = 4,
        dropout_rate: float = 0,
        classifier: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.densenet = self._configure_densenet(
            in_channels,
            base_channels,
            num_classes,
            growth_rate,
            block_config,
            bn_size,
            dropout_rate,
            classifier,
            activation,
        )

    def _configure_densenet(
        self,
        in_channels: int,
        base_channels: int,
        num_classes: int,
        growth_rate: int,
        block_config: List[int],
        bn_size: int,
        dropout_rate: float,
        classifier: bool,
        activation: str,
    ) -> nn.Sequential:
        activation_fn = activation_function(activation)
        densenet = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            activation_fn,
        ]

        num_features = base_channels

        for i, num_layers in enumerate(block_config):
            densenet.append(
                _DenseBlock(
                    num_layers=num_layers,
                    in_channels=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    dropout_rate=dropout_rate,
                    activation=activation,
                )
            )
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                densenet.append(
                    _Transition(
                        in_channels=num_features,
                        out_channels=num_features // 2,
                        activation=activation,
                    )
                )
                num_features = num_features // 2

        densenet.append(activation_fn)

        if classifier:
            densenet.append(nn.AdaptiveAvgPool2d((1, 1)))
            densenet.append(Rearrange("b c h w -> b (c h w)"))
            densenet.append(
                nn.Linear(in_features=num_features, out_features=num_classes)
            )

        return nn.Sequential(*densenet)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of Densenet."""
        # If batch dimenstion is missing, it will be added.
        if len(x.shape) < 4:
            x = x[(None,) * (4 - len(x.shape))]
        return self.densenet(x)

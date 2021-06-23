"""Mobile inverted residual block."""
from typing import Any, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .utils import calculate_output_image_size, drop_connection, get_same_padding_conv2d


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bn_momentum: float,
        bn_eps: float,
        se_ratio: float,
        expand_ratio: int,
        image_size: Optional[Tuple[int, int]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.in_channels = self.in_channels
        self.out_channels = out_channels

        # Placeholders for layers.
        self._inverted_bottleneck: nn.Sequential = None
        self._depthwise: nn.Sequential = None
        self._squeeze_excite: nn.Sequential = None
        self._pointwise: nn.Sequential = None

        self._build(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expand_ratio=expand_ratio,
            se_ratio=se_ratio,
        )

    def _build(
        self,
        image_size: Optional[Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
    ) -> None:
        has_se = se_ratio is not None and 0.0 < se_ratio < 1.0
        inner_channels = in_channels * expand_ratio
        self._inverted_bottleneck = (
            self._configure_inverted_bottleneck(
                image_size=image_size,
                in_channels=in_channels,
                out_channels=inner_channels,
            )
            if expand_ratio != 1
            else None
        )

        self._depthwise = self._configure_depthwise(
            image_size=image_size,
            in_channels=inner_channels,
            out_channels=inner_channels,
            groups=inner_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        image_size = calculate_output_image_size(image_size, stride)
        self._squeeze_excite = (
            self._configure_squeeze_excite(
                in_channels=inner_channels,
                out_channels=inner_channels,
                se_ratio=se_ratio,
            )
            if has_se
            else None
        )

        self._pointwise = self._configure_pointwise(
            image_size=image_size, in_channels=inner_channels, out_channels=out_channels
        )

    def _configure_inverted_bottleneck(
        self,
        image_size: Optional[Tuple[int, int]],
        in_channels: int,
        out_channels: int,
    ) -> nn.Sequential:
        """Expansion phase."""
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        return nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels, momentum=self.bn_momentum, eps=self.bn_eps
            ),
            nn.Mish(inplace=True),
        )

    def _configure_depthwise(
        self,
        image_size: Optional[Tuple[int, int]],
        in_channels: int,
        out_channels: int,
        groups: int,
        kernel_size: int,
        stride: int,
    ) -> nn.Sequential:
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        return nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels, momentum=self.bn_momentum, eps=self.bn_eps
            ),
            nn.Mish(inplace=True),
        )

    def _configure_squeeze_excite(
        self, in_channels: int, out_channels: int, se_ratio: float
    ) -> nn.Sequential:
        Conv2d = get_same_padding_conv2d(image_size=(1, 1))
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        return nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
            ),
            nn.Mish(inplace=True),
            Conv2d(
                in_channels=num_squeezed_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def _configure_pointwise(
        self, image_size: Optional[Tuple[int, int]], in_channels: int, out_channels: int
    ) -> nn.Sequential:
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        return nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels, momentum=self.bn_momentum, eps=self.bn_eps
            ),
        )

    def _stochastic_depth(
        self, x: Tensor, residual: Tensor, drop_connection_rate: Optional[float]
    ) -> Tensor:
        if self.id_skip and self.stride == 1 and self.in_channels == self.out_channels:
            if drop_connection_rate:
                x = drop_connection(x, p=drop_connection_rate, training=self.training)
            x += residual
        return x

    def forward(
        self, x: Tensor, drop_connection_rate: Optional[float] = None
    ) -> Tensor:
        residual = x
        if self._inverted_bottleneck is not None:
            x = self._inverted_bottleneck(x)

        x = self._depthwise(x)

        if self._squeeze_excite is not None:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._squeeze_excite(x)
            x = torch.tanh(F.softplus(x_squeezed)) * x

        x = self._pointwise(x)

        # Stochastic depth
        x = self._stochastic_depth(x, residual, drop_connection_rate)
        return x

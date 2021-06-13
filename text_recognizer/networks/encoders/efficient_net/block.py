"""Mobile inverted residual block."""
from typing import Tuple

import torch
from torch import nn, Tensor

from .utils import get_same_padding_conv2d


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        bn_momentum: float,
        bn_eps: float,
        se_ratio: float,
        id_skip: bool,
        expand_ratio: int,
        image_size: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.has_se = se_ratio is not None and 0.0 < se_ratio < 1.0

        out_channels = in_channels * expand_ratio
        self._inverted_bottleneck = (
            self._configure_inverted_bottleneck(
                image_size=image_size,
                in_channels=in_channels,
                out_channels=out_channels,
            )
            if expand_ratio != 1
            else None
        )

        self._depthwise = self._configure_depthwise(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            groups=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        image_size = calculate_output_image_size(image_size, stride)
        self._squeeze_excite = (
            self._configure_squeeze_excite(
                in_channels=out_channels, out_channels=out_channels, se_ratio=se_ratio
            )
            if self.has_se
            else None
        )

        self._pointwise = self._configure_pointwise(
            image_size=image_size, in_channels=out_channels, out_channels=out_channels
        )

    def _configure_inverted_bottleneck(
        self,
        image_size: Tuple[int, int],
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
            nn.SiLU(inplace=True),
        )

    def _configure_depthwise(
        self,
        image_size: Tuple[int, int],
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
            nn.SiLU(inplace=True),
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
            nn.SiLU(inplace=True),
            Conv2d(
                in_channels=num_squeezed_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def _configure_pointwise(
        self, image_size: Tuple[int, int], in_channels: int, out_channels: int
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

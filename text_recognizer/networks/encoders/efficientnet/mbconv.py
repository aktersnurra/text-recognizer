"""Mobile inverted residual block."""
from typing import Any, Optional, Union, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .utils import stochastic_depth


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple[int, int], int],
        bn_momentum: float,
        bn_eps: float,
        se_ratio: float,
        expand_ratio: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = (stride,) * 2 if isinstance(stride, int) else stride
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.stride == (2, 2):
            self.pad = [
                (self.kernel_size - 1) // 2 - 1,
                (self.kernel_size - 1) // 2,
            ] * 2
        else:
            self.pad = [(self.kernel_size - 1) // 2] * 4

        # Placeholders for layers.
        self._inverted_bottleneck: nn.Sequential = None
        self._depthwise: nn.Sequential = None
        self._squeeze_excite: nn.Sequential = None
        self._pointwise: nn.Sequential = None

        self._build(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expand_ratio=expand_ratio,
            se_ratio=se_ratio,
        )

    def _build(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple[int, int], int],
        expand_ratio: int,
        se_ratio: float,
    ) -> None:
        has_se = se_ratio is not None and 0.0 < se_ratio < 1.0
        inner_channels = in_channels * expand_ratio
        self._inverted_bottleneck = (
            self._configure_inverted_bottleneck(
                in_channels=in_channels, out_channels=inner_channels,
            )
            if expand_ratio != 1
            else None
        )

        self._depthwise = self._configure_depthwise(
            in_channels=inner_channels,
            out_channels=inner_channels,
            groups=inner_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

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
            in_channels=inner_channels, out_channels=out_channels
        )

    def _configure_inverted_bottleneck(
        self, in_channels: int, out_channels: int,
    ) -> nn.Sequential:
        """Expansion phase."""
        return nn.Sequential(
            nn.Conv2d(
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
        in_channels: int,
        out_channels: int,
        groups: int,
        kernel_size: int,
        stride: Union[Tuple[int, int], int],
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
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
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
            ),
            nn.Mish(inplace=True),
            nn.Conv2d(
                in_channels=num_squeezed_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def _configure_pointwise(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
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
        self, x: Tensor, residual: Tensor, stochastic_dropout_rate: Optional[float]
    ) -> Tensor:
        if self.stride == (1, 1) and self.in_channels == self.out_channels:
            if stochastic_dropout_rate:
                x = stochastic_depth(
                    x, p=stochastic_dropout_rate, training=self.training
                )
            x += residual
        return x

    def forward(
        self, x: Tensor, stochastic_dropout_rate: Optional[float] = None
    ) -> Tensor:
        residual = x
        if self._inverted_bottleneck is not None:
            x = self._inverted_bottleneck(x)
        x = F.pad(x, self.pad)

        x = self._depthwise(x)

        if self._squeeze_excite is not None:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._squeeze_excite(x)
            x = torch.tanh(F.softplus(x_squeezed)) * x

        x = self._pointwise(x)

        # Stochastic depth
        x = self._stochastic_depth(x, residual, stochastic_dropout_rate)
        return x

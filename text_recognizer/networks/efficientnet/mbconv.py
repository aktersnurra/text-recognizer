"""Mobile inverted residual block."""
from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from text_recognizer.networks.efficientnet.utils import stochastic_depth


def _convert_stride(stride: Union[Tuple[int, int], int]) -> Tuple[int, int]:
    """Converts int to tuple."""
    return (stride,) * 2 if isinstance(stride, int) else stride


class BaseModule(nn.Module):
    """Base sub module class."""

    def __init__(self, bn_momentum: float, bn_eps: float, block: nn.Sequential) -> None:
        super().__init__()

        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.block = block
        self._build()

    def _build(self) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.block(x)


class InvertedBottleneck(BaseModule):
    """Inverted bottleneck module."""

    def __init__(
        self,
        bn_momentum: float,
        bn_eps: float,
        block: nn.Sequential,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(bn_momentum, bn_eps, block)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _build(self) -> None:
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=self.out_channels,
                momentum=self.bn_momentum,
                eps=self.bn_eps,
            ),
            nn.Mish(inplace=True),
        )


class Depthwise(BaseModule):
    """Depthwise convolution module."""

    def __init__(
        self,
        bn_momentum: float,
        bn_eps: float,
        block: nn.Sequential,
        channels: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__(bn_momentum, bn_eps, block)
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

    def _build(self) -> None:
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.channels,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=self.channels, momentum=self.bn_momentum, eps=self.bn_eps
            ),
            nn.Mish(inplace=True),
        )


class SqueezeAndExcite(BaseModule):
    """Sequeeze and excite module."""

    def __init__(
        self,
        bn_momentum: float,
        bn_eps: float,
        block: nn.Sequential,
        in_channels: int,
        channels: int,
        se_ratio: float,
    ) -> None:
        super().__init__(bn_momentum, bn_eps, block)

        self.in_channels = in_channels
        self.channels = channels
        self.se_ratio = se_ratio

    def _build(self) -> None:
        num_squeezed_channels = max(1, int(self.in_channels * self.se_ratio))
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
            ),
            nn.Mish(inplace=True),
            nn.Conv2d(
                in_channels=num_squeezed_channels,
                out_channels=self.channels,
                kernel_size=1,
            ),
        )


class Pointwise(BaseModule):
    """Pointwise module."""

    def __init__(
        self,
        bn_momentum: float,
        bn_eps: float,
        block: nn.Sequential,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(bn_momentum, bn_eps, block)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _build(self) -> None:
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=self.out_channels,
                momentum=self.bn_momentum,
                eps=self.bn_eps,
            ),
        )


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        bn_momentum: float,
        bn_eps: float,
        se_ratio: float,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.se_ratio = se_ratio
        self.expand_ratio = expand_ratio
        self.pad = self._configure_padding()
        self._inverted_bottleneck: Optional[InvertedBottleneck]
        self._depthwise: nn.Sequential
        self._squeeze_excite: nn.Sequential
        self._pointwise: nn.Sequential
        self._build()

    def _configure_padding(self) -> Tuple[int, int, int, int]:
        """Set padding for convolutional layers."""
        if self.stride == (2, 2):
            return (
                (self.kernel_size - 1) // 2 - 1,
                (self.kernel_size - 1) // 2,
            ) * 2
        return ((self.kernel_size - 1) // 2,) * 4

    def _build(self) -> None:
        has_se = self.se_ratio is not None and 0.0 < self.se_ratio < 1.0
        inner_channels = self.in_channels * self.expand_ratio
        self._inverted_bottleneck = (
            InvertedBottleneck(
                in_channels=self.in_channels,
                out_channels=inner_channels,
                bn_momentum=self.bn_momentum,
                bn_eps=self.bn_eps,
            )
            if self.expand_ratio != 1
            else None
        )

        self._depthwise = Depthwise(
            channels=inner_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bn_momentum=self.bn_momentum,
            bn_eps=self.bn_eps,
        )

        self._squeeze_excite = (
            SqueezeAndExcite(
                in_channels=self.in_channels,
                channels=inner_channels,
                se_ratio=self.se_ratio,
                bn_momentum=self.bn_momentum,
                bn_eps=self.bn_eps,
            )
            if has_se
            else None
        )

        self._pointwise = Pointwise(
            in_channels=inner_channels,
            out_channels=self.out_channels,
            bn_momentum=self.bn_momentum,
            bn_eps=self.bn_eps,
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
        """Forward pass."""
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

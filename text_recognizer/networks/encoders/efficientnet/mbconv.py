"""Mobile inverted residual block."""
from typing import Optional, Sequence, Union, Tuple

import attr
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from text_recognizer.networks.encoders.efficientnet.utils import stochastic_depth


def _convert_stride(stride: Union[Tuple[int, int], int]) -> Tuple[int, int]:
    """Converts int to tuple."""
    return (
        (stride,) * 2 if isinstance(stride, int) else stride
    )


@attr.s(eq=False)
class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck block."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    in_channels: int = attr.ib()
    out_channels: int = attr.ib()
    kernel_size: Tuple[int, int] = attr.ib()
    stride: Tuple[int, int] = attr.ib(converter=_convert_stride)
    bn_momentum: float = attr.ib()
    bn_eps: float = attr.ib()
    se_ratio: float = attr.ib()
    expand_ratio: int = attr.ib()
    pad: Tuple[int, int, int, int] = attr.ib(init=False)
    _inverted_bottleneck: nn.Sequential = attr.ib(init=False)
    _depthwise: nn.Sequential = attr.ib(init=False)
    _squeeze_excite: nn.Sequential = attr.ib(init=False)
    _pointwise: nn.Sequential = attr.ib(init=False)

    @pad.default
    def _configure_padding(self) -> Tuple[int, int, int, int]:
        """Set padding for convolutional layers."""
        if self.stride == (2, 2):
            return (
                (self.kernel_size - 1) // 2 - 1,
                (self.kernel_size - 1) // 2,
                ) * 2
        return ((self.kernel_size - 1) // 2,) * 4

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        self._build()

    def _build(self) -> None:
        has_se = self.se_ratio is not None and 0.0 < self.se_ratio < 1.0
        inner_channels = self.in_channels * self.expand_ratio
        self._inverted_bottleneck = (
            self._configure_inverted_bottleneck(out_channels=inner_channels)
            if self.expand_ratio != 1
            else None
        )

        self._depthwise = self._configure_depthwise(
            in_channels=inner_channels,
            out_channels=inner_channels,
            groups=inner_channels,
        )

        self._squeeze_excite = (
            self._configure_squeeze_excite(
                in_channels=inner_channels, out_channels=inner_channels,
            )
            if has_se
            else None
        )

        self._pointwise = self._configure_pointwise(in_channels=inner_channels)

    def _configure_inverted_bottleneck(self, out_channels: int) -> nn.Sequential:
        """Expansion phase."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
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
        self, in_channels: int, out_channels: int, groups: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels, momentum=self.bn_momentum, eps=self.bn_eps
            ),
            nn.Mish(inplace=True),
        )

    def _configure_squeeze_excite(
        self, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        num_squeezed_channels = max(1, int(in_channels * self.se_ratio))
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

    def _configure_pointwise(self, in_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
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

"""Efficientnet backbone."""
from typing import Tuple

import attr
from torch import nn, Tensor

from text_recognizer.networks.efficientnet.mbconv import MBConvBlock
from text_recognizer.networks.efficientnet.utils import (
    block_args,
    round_filters,
    round_repeats,
)


@attr.s(eq=False)
class EfficientNet(nn.Module):
    """Efficientnet without classification head."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    archs = {
        # width, depth, dropout
        "b0": (1.0, 1.0, 0.2),
        "b1": (1.0, 1.1, 0.2),
        "b2": (1.1, 1.2, 0.3),
        "b3": (1.2, 1.4, 0.3),
        "b4": (1.4, 1.8, 0.4),
        "b5": (1.6, 2.2, 0.4),
        "b6": (1.8, 2.6, 0.5),
        "b7": (2.0, 3.1, 0.5),
        "b8": (2.2, 3.6, 0.5),
        "l2": (4.3, 5.3, 0.5),
    }

    arch: str = attr.ib()
    params: Tuple[float, float, float] = attr.ib(default=None, init=False)
    stochastic_dropout_rate: float = attr.ib(default=0.2)
    bn_momentum: float = attr.ib(default=0.99)
    bn_eps: float = attr.ib(default=1.0e-3)
    depth: int = attr.ib(default=7)
    out_channels: int = attr.ib(default=None, init=False)
    _conv_stem: nn.Sequential = attr.ib(default=None, init=False)
    _blocks: nn.ModuleList = attr.ib(default=None, init=False)
    _conv_head: nn.Sequential = attr.ib(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        self._build()

    @depth.validator
    def _check_depth(self, attribute: attr._make.Attribute, value: str) -> None:
        if not 5 <= value <= 7:
            raise ValueError(f"Depth has to be between 5 and 7, was: {value}")

    @arch.validator
    def _check_arch(self, attribute: attr._make.Attribute, value: str) -> None:
        """Validates the efficientnet architecure."""
        if value not in self.archs:
            raise ValueError(f"{value} not a valid architecure.")
        self.params = self.archs[value]

    def _build(self) -> None:
        """Builds the efficientnet backbone."""
        _block_args = block_args()[: self.depth]
        in_channels = 1  # BW
        out_channels = round_filters(32, self.params)
        self._conv_stem = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=(2, 2),
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=out_channels, momentum=self.bn_momentum, eps=self.bn_eps
            ),
            nn.Mish(inplace=True),
        )
        self._blocks = nn.ModuleList([])
        for args in _block_args:
            args.in_channels = round_filters(args.in_channels, self.params)
            args.out_channels = round_filters(args.out_channels, self.params)
            num_repeats = round_repeats(args.num_repeats, self.params)
            del args.num_repeats
            for _ in range(num_repeats):
                self._blocks.append(
                    MBConvBlock(
                        **args, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps,
                    )
                )
                args.in_channels = args.out_channels
                args.stride = 1

        in_channels = round_filters(_block_args[-1].out_channels, self.params)
        self.out_channels = round_filters(1280, self.params)
        self._conv_head = nn.Sequential(
            nn.Conv2d(
                in_channels, self.out_channels, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(
                num_features=self.out_channels,
                momentum=self.bn_momentum,
                eps=self.bn_eps,
            ),
            nn.Dropout(p=self.params[-1]),
        )

    def extract_features(self, x: Tensor) -> Tensor:
        """Extracts the final feature map layer."""
        x = self._conv_stem(x)
        for i, block in enumerate(self._blocks):
            stochastic_dropout_rate = self.stochastic_dropout_rate
            if self.stochastic_dropout_rate:
                stochastic_dropout_rate *= i / len(self._blocks)
            x = block(x, stochastic_dropout_rate=stochastic_dropout_rate)
        x = self._conv_head(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Returns efficientnet image features."""
        return self.extract_features(x)
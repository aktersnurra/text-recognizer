"""Efficient net."""
from torch import nn, Tensor

from .mbconv import MBConvBlock
from .utils import (
    block_args,
    round_filters,
    round_repeats,
)


class EfficientNet(nn.Module):
    archs = {
        #     width,depth0res,dropout
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

    def __init__(
        self,
        arch: str,
        out_channels: int = 1280,
        stochastic_dropout_rate: float = 0.2,
        bn_momentum: float = 0.99,
        bn_eps: float = 1.0e-3,
    ) -> None:
        super().__init__()
        assert arch in self.archs, f"{arch} not a valid efficient net architecure!"
        self.arch = self.archs[arch]
        self.out_channels = out_channels
        self.stochastic_dropout_rate = stochastic_dropout_rate
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self._conv_stem: nn.Sequential = None
        self._blocks: nn.Sequential = None
        self._conv_head: nn.Sequential = None
        self._build()

    def _build(self) -> None:
        _block_args = block_args()
        in_channels = 1  # BW
        out_channels = round_filters(32, self.arch)
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
            args.in_channels = round_filters(args.in_channels, self.arch)
            args.out_channels = round_filters(args.out_channels, self.arch)
            args.num_repeats = round_repeats(args.num_repeats, self.arch)
            for _ in range(args.num_repeats):
                self._blocks.append(
                    MBConvBlock(
                        **args, bn_momentum=self.bn_momentum, bn_eps=self.bn_eps,
                    )
                )
                args.in_channels = args.out_channels
                args.stride = 1

        in_channels = round_filters(320, self.arch)
        out_channels = round_filters(self.out_channels, self.arch)
        self._conv_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(
                num_features=out_channels, momentum=self.bn_momentum, eps=self.bn_eps
            ),
        )

    def extract_features(self, x: Tensor) -> Tensor:
        x = self._conv_stem(x)
        for i, block in enumerate(self._blocks):
            stochastic_dropout_rate = self.stochastic_dropout_rate
            if self.stochastic_dropout_rate:
                stochastic_dropout_rate *= i / len(self._blocks)
            x = block(x, stochastic_dropout_rate=stochastic_dropout_rate)
        x = self._conv_head(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.extract_features(x)

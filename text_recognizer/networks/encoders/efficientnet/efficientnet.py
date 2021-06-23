"""Efficient net."""
from typing import Tuple

from torch import nn, Tensor

from .mbconv import MBConvBlock
from .utils import (
    block_args,
    calculate_output_image_size,
    get_same_padding_conv2d,
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

    def __init__(self, arch: str, image_size: Tuple[int, int]) -> None:
        super().__init__()
        assert arch in self.archs, f"{arch} not a valid efficient net architecure!"
        self.arch = self.archs[arch]
        self.image_size = image_size
        self._conv_stem: nn.Sequential = None
        self._blocks: nn.Sequential = None
        self._conv_head: nn.Sequential = None
        self._build()

    def _build(self) -> None:
        _block_args = block_args()
        in_channels = 1  # BW
        out_channels = round_filters(32, self.arch)
        Conv2d = get_same_padding_conv2d(image_size=self.image_size)
        self._conv_stem = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels, momentum=bn_momentum, eps=bn_eps),
            nn.Mish(inplace=True),
        )
        image_size = calculate_output_image_size(self.image_size, 2)
        self._blocks = nn.ModuleList([])
        for args in _block_args:
            args.in_channels = round_filters(args.in_channels, self.arch)
            args.out_channels = round_filters(args.out_channels, self.arch)
            args.num_repeat = round_repeats(args.num_repeat, self.arch)

            self._blocks.append(
                MBConvBlock(
                    **args,
                    bn_momentum=bn_momentum,
                    bn_eps=bn_eps,
                    image_size=image_size,
                )
            )
            image_size = calculate_output_image_size(image_size, args.stride)
            if args.num_repeat > 1:
                args.in_channels = args.out_channels
                args.stride = 1
            for _ in range(args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(
                        **args,
                        bn_momentum=bn_momentum,
                        bn_eps=bn_eps,
                        image_size=image_size,
                    )
                )

        in_channels = args.out_channels
        out_channels = round_filters(1280, self.arch)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=bn_momentum, eps=bn_eps),
        )

    def extract_features(self, x: Tensor) -> Tensor:
        x = self._conv_stem(x)

    def forward(self, x: Tensor) -> Tensor:
        pass

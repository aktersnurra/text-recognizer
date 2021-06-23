"""Util functions for efficient net."""
from functools import partial
import math
from typing import Any, Optional, Tuple, Type

from omegaconf import OmegaConf
import torch
from torch import nn, Tensor
import torch.functional as F


def calculate_output_image_size(
    image_size: Optional[Tuple[int, int]], stride: int
) -> Optional[Tuple[int, int]]:
    """Calculates the output image size when using conv2d with same padding."""
    if image_size is None:
        return None
    height = int(math.ceil(image_size[0] / stride))
    width = int(math.ceil(image_size[1] / stride))
    return height, width


def drop_connection(x: Tensor, p: float, training: bool) -> Tensor:
    """Drop connection.

    Drops the entire convolution with a given survival probability.

    Args:
        x (Tensor): Input tensor.
        p (float): Survival probability between 0.0 and 1.0.
        training (bool): The running mode.

    Shapes:
        - x: :math: `(B, C, W, H)`.
        - out: :math: `(B, C, W, H)`.

        where B is the batch size, C is the number of channels, W is the width, and H
        is the height.

    Returns:
        out (Tensor): Output after drop connection.
    """
    assert 0.0 <= p <= 1.0, "p must be in range of [0, 1]"

    if not training:
        return x

    bsz = x.shape[0]
    survival_prob = 1 - p

    # Generate a binary tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = survival_prob
    random_tensor += torch.rand([bsz, 1, 1, 1]).type_as(x)
    binary_tensor = torch.floor(random_tensor)

    out = x / survival_prob * binary_tensor
    return out


def get_same_padding_conv2d(image_size: Optional[Tuple[int, int]]) -> Type[nn.Conv2d]:
    if image_size is None:
        return Conv2dDynamicSamePadding
    return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        self.stride = [self.stride] * 2

    def forward(self, x: Tensor) -> Tensor:
        ih, iw = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        image_size: Tuple[int, int],
        stride: int = 1,
        **kwargs: Any
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2

        # Calculate padding based on image size and save it.
        ih, iw = image_size
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.static_padding(x)
        x = F.pad(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


def round_filters(filters: int, arch: Tuple[float, float, float]) -> int:
    multiplier = arch[0]
    divisor = 8
    filters *= multiplier
    new_filters = max(divisor, (filters + divisor // 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, arch: Tuple[float, float, float]) -> int:
    return int(math.ceil(arch[1] * repeats))


def block_args():
    keys = [
        "num_repeats",
        "kernel_size",
        "strides",
        "expand_ratio",
        "in_channels",
        "out_channels",
        "se_ratio",
    ]
    args = [
        [1, 3, (1, 1), 1, 32, 16, 0.25],
        [2, 3, (2, 2), 6, 16, 24, 0.25],
        [2, 5, (2, 2), 6, 24, 40, 0.25],
        [3, 3, (2, 2), 6, 40, 80, 0.25],
        [3, 5, (1, 1), 6, 80, 112, 0.25],
        [4, 5, (2, 2), 6, 112, 192, 0.25],
        [1, 3, (1, 1), 6, 192, 320, 0.25],
    ]
    block_args_ = []
    for row in args:
        block_args_.append(OmegaConf.create(dict(zip(keys, row))))
    return block_args_

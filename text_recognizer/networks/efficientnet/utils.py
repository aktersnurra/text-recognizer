"""Util functions for efficient net."""
import math
from typing import List, Tuple

from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor


def stochastic_depth(x: Tensor, p: float, training: bool) -> Tensor:
    """Stochastic connection.

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


def round_filters(filters: int, arch: Tuple[float, float, float]) -> int:
    """Returns the number output filters for a block."""
    multiplier = arch[0]
    divisor = 8
    filters *= multiplier
    new_filters = max(divisor, (filters + divisor // 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, arch: Tuple[float, float, float]) -> int:
    """Returns how many times a layer should be repeated in a block."""
    return int(math.ceil(arch[1] * repeats))


def block_args() -> List[DictConfig]:
    """Returns arguments for each efficientnet block."""
    keys = [
        "num_repeats",
        "kernel_size",
        "stride",
        "expand_ratio",
        "in_channels",
        "out_channels",
        "se_ratio",
    ]
    args = [
        [1, 3, (1, 1), 1, 16, 16, 0.25],
        [2, 3, (2, 2), 6, 16, 24, 0.25],
        [2, 5, (2, 2), 6, 24, 40, 0.25],
        [3, 3, (2, 1), 6, 40, 80, 0.25],
        [3, 5, (2, 1), 6, 80, 112, 0.25],
        [4, 5, (1, 1), 6, 112, 192, 0.25],
        [1, 3, (2, 1), 6, 192, 320, 0.25],
    ]
    block_args_ = []
    for row in args:
        block_args_.append(OmegaConf.create(dict(zip(keys, row))))
    return block_args_

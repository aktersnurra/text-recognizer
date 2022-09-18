"""Pad targets to equal length."""

import torch
import torch.functional as F
from torch import Tensor


class Pad:
    """Pad target sequence."""

    def __init__(self, max_len: int, pad_index: int) -> None:
        self.max_len = max_len
        self.pad_index = pad_index

    def __call__(self, y: Tensor) -> Tensor:
        """Pads sequences with pad index if shorter than max len."""
        if y.shape[-1] < self.max_len:
            pad_len = self.max_len - len(y)
            y = torch.cat((y, torch.LongTensor([self.pad_index] * pad_len)))
        return y[: self.max_len]

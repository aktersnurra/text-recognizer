"""Efficient net."""
from torch import nn, Tensor


class EfficientNet(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

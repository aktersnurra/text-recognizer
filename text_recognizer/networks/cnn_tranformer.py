"""Vision transformer for character recognition."""
from typing import Type

import attr
from torch import nn, Tensor


@attr.s
class CnnTransformer(nn.Module):
    def __attrs_pre_init__(self) -> None:
        super().__init__()

    backbone: Type[nn.Module] = attr.ib()
    head = Type[nn.Module] = attr.ib()

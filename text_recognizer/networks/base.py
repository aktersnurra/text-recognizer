"""Base network with required methods."""
from abc import abstractmethod

import attr
from torch import nn, Tensor


@attr.s
class BaseNetwork(nn.Module):
    """Base network."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Return token indices for predictions."""
        ...

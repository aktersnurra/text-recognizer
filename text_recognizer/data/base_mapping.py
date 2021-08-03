"""Mapping to and from word pieces."""
from abc import ABC, abstractmethod
from typing import Dict, List

from torch import Tensor


class AbstractMapping(ABC):
    def __init__(
        self, input_size: List[int], mapping: List[str], inverse_mapping: Dict[str, int]
    ) -> None:
        self.input_size = input_size
        self.mapping = mapping
        self.inverse_mapping = inverse_mapping

    def __len__(self) -> int:
        return len(self.mapping)

    @property
    def num_classes(self) -> int:
        return self.__len__()

    @abstractmethod
    def get_token(self, *args, **kwargs) -> str:
        ...

    @abstractmethod
    def get_index(self, *args, **kwargs) -> Tensor:
        ...

    @abstractmethod
    def get_text(self, *args, **kwargs) -> str:
        ...

    @abstractmethod
    def get_indices(self, *args, **kwargs) -> Tensor:
        ...

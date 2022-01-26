"""Emnist mapping."""
from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor

from text_recognizer.data.emnist import emnist_mapping
from text_recognizer.data.mappings.base import AbstractMapping


class EmnistMapping(AbstractMapping):
    """Mapping for EMNIST labels."""

    def __init__(
        self, extra_symbols: Optional[Sequence[str]] = None, lower: bool = True
    ) -> None:
        self.extra_symbols = set(extra_symbols) if extra_symbols is not None else None
        self.mapping, self.inverse_mapping, self.input_size = emnist_mapping(
            self.extra_symbols
        )
        if lower:
            self._to_lower()
        super().__init__(self.input_size, self.mapping, self.inverse_mapping)

    def _to_lower(self) -> None:
        """Converts mapping to lowercase letters only."""

        def _filter(x: int) -> int:
            if 40 <= x:
                return x - 26
            return x

        self.inverse_mapping = {v: _filter(k) for k, v in enumerate(self.mapping)}
        self.mapping = [c for c in self.mapping if not c.isupper()]

    def get_token(self, index: Union[int, Tensor]) -> str:
        """Returns token for index value."""
        if (index := int(index)) <= len(self.mapping):
            return self.mapping[index]
        raise KeyError(f"Index ({index}) not in mapping.")

    def get_index(self, token: str) -> Tensor:
        """Returns index value of token."""
        if token in self.inverse_mapping:
            return torch.LongTensor([self.inverse_mapping[token]])
        raise KeyError(f"Token ({token}) not found in inverse mapping.")

    def get_text(self, indices: Union[List[int], Tensor]) -> str:
        """Returns the text from a list of indices."""
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        return "".join([self.mapping[index] for index in indices])

    def get_indices(self, text: str) -> Tensor:
        """Returns tensor of indices for a string."""
        return Tensor([self.inverse_mapping[token] for token in text])

    def __getitem__(self, x: Union[int, Tensor]) -> str:
        """Returns text for a list of indices."""
        return self.get_token(x)

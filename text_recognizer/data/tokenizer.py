"""Emnist mapping."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

import text_recognizer.metadata.shared as metadata


class Tokenizer:
    """Mapping for EMNIST labels."""

    def __init__(
        self,
        extra_symbols: Optional[Sequence[str]] = None,
        lower: bool = True,
        start_token: str = "<s>",
        end_token: str = "<e>",
        pad_token: str = "<p>",
    ) -> None:
        self.extra_symbols = set(extra_symbols) if extra_symbols is not None else None
        self.mapping, self.inverse_mapping, self.input_size = self._load_mapping()
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.start_index = int(self.get_value(self.start_token))
        self.end_index = int(self.get_value(self.end_token))
        self.pad_index = int(self.get_value(self.pad_token))
        self.ignore_indices = set([self.start_index, self.end_index, self.pad_index])
        if lower:
            self._to_lower()

    def __len__(self) -> int:
        return len(self.mapping)

    @property
    def num_classes(self) -> int:
        """Return number of classes in the dataset."""
        return self.__len__()

    def _load_mapping(self) -> Tuple[List, Dict[str, int], List[int]]:
        """Return the EMNIST mapping."""
        with metadata.ESSENTIALS_FILENAME.open() as f:
            essentials = json.load(f)
        mapping = list(essentials["characters"])
        if self.extra_symbols is not None:
            mapping += self.extra_symbols
        inverse_mapping = {v: k for k, v in enumerate(mapping)}
        input_shape = essentials["input_shape"]
        return mapping, inverse_mapping, input_shape

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

    def get_value(self, token: str) -> Tensor:
        """Returns index value of token."""
        if token in self.inverse_mapping:
            return torch.LongTensor([self.inverse_mapping[token]])
        raise KeyError(f"Token ({token}) not found in inverse mapping.")

    def decode(self, indices: Union[List[int], Tensor]) -> str:
        """Returns the text from a list of indices."""
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        return "".join(
            [
                self.mapping[index]
                for index in indices
                if index not in self.ignore_indices
            ]
        )

    def encode(self, text: str) -> Tensor:
        """Returns tensor of indices for a string."""
        return Tensor([self.inverse_mapping[token] for token in text])

    def __getitem__(self, x: Union[int, Tensor]) -> str:
        """Returns text for a list of indices."""
        return self.get_token(x)

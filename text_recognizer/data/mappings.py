"""Mapping to and from word pieces."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Sequence

import attr
import loguru.logger as log
import torch
from torch import Tensor

from text_recognizer.data.emnist import emnist_mapping
from text_recognizer.data.iam_preprocessor import Preprocessor


class AbstractMapping(ABC):
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


@attr.s
class EmnistMapping(AbstractMapping):
    extra_symbols: Optional[Set[str]] = attr.ib(default=None, converter=set)
    mapping: Sequence[str] = attr.ib(init=False)
    inverse_mapping: Dict[str, int] = attr.ib(init=False)
    input_size: List[int] = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        self.mapping, self.inverse_mapping, self.input_size = emnist_mapping(
            self.extra_symbols
        )

    def get_token(self, index: Union[int, Tensor]) -> str:
        if (index := int(index)) in self.mapping:
            return self.mapping[index]
        raise KeyError(f"Index ({index}) not in mapping.")

    def get_index(self, token: str) -> Tensor:
        if token in self.inverse_mapping:
            return Tensor(self.inverse_mapping[token])
        raise KeyError(f"Token ({token}) not found in inverse mapping.")

    def get_text(self, indices: Union[List[int], Tensor]) -> str:
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        return "".join([self.mapping[index] for index in indices])

    def get_indices(self, text: str) -> Tensor:
        return Tensor([self.inverse_mapping[token] for token in text])


@attr.s(auto_attribs=True)
class WordPieceMapping(EmnistMapping):
    data_dir: Optional[Path] = attr.ib(default=None)
    num_features: int = attr.ib(default=1000)
    tokens: str = attr.ib(default="iamdb_1kwp_tokens_1000.txt")
    lexicon: str = attr.ib(default="iamdb_1kwp_lex_1000.txt")
    use_words: bool = attr.ib(default=False)
    prepend_wordsep: bool = attr.ib(default=False)
    special_tokens: Set[str] = attr.ib(default={"<s>", "<e>", "<p>"}, converter=set)
    extra_symbols: Set[str] = attr.ib(default={"\n",}, converter=set)
    wordpiece_processor: Preprocessor = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        super().__attrs_post_init__()
        self.data_dir = (
            (
                Path(__file__).resolve().parents[2]
                / "data"
                / "downloaded"
                / "iam"
                / "iamdb"
            )
            if self.data_dir is None
            else Path(self.data_dir)
        )
        log.debug(f"Using data dir: {self.data_dir}")
        if not self.data_dir.exists():
            raise RuntimeError(f"Could not locate iamdb directory at {self.data_dir}")

        processed_path = (
            Path(__file__).resolve().parents[2] / "data" / "processed" / "iam_lines"
        )

        tokens_path = processed_path / self.tokens
        lexicon_path = processed_path / self.lexicon

        special_tokens = self.special_tokens
        if self.extra_symbols is not None:
            special_tokens = special_tokens | self.extra_symbols

        self.wordpiece_processor = Preprocessor(
            data_dir=self.data_dir,
            num_features=self.num_features,
            tokens_path=tokens_path,
            lexicon_path=lexicon_path,
            use_words=self.use_words,
            prepend_wordsep=self.prepend_wordsep,
            special_tokens=special_tokens,
        )

    def __len__(self) -> int:
        return len(self.wordpiece_processor.tokens)

    def get_token(self, index: Union[int, Tensor]) -> str:
        if (index := int(index)) <= self.wordpiece_processor.num_tokens:
            return self.wordpiece_processor.tokens[index]
        raise KeyError(f"Index ({index}) not in mapping.")

    def get_index(self, token: str) -> Tensor:
        if token in self.wordpiece_processor.tokens:
            return torch.LongTensor([self.wordpiece_processor.tokens_to_index[token]])
        raise KeyError(f"Token ({token}) not found in inverse mapping.")

    def get_text(self, indices: Union[List[int], Tensor]) -> str:
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        return self.wordpiece_processor.to_text(indices)

    def get_indices(self, text: str) -> Tensor:
        return self.wordpiece_processor.to_index(text)

    def emnist_to_wordpiece_indices(self, x: Tensor) -> Tensor:
        text = "".join([self.mapping[i] for i in x])
        text = text.lower().replace(" ", "▁")
        return torch.LongTensor(self.wordpiece_processor.to_index(text))

    def __getitem__(self, x: Union[str, int, List[int], Tensor]) -> Union[str, Tensor]:
        if isinstance(x, int):
            x = [x]
        if isinstance(x, str):
            return self.get_indices(x)
        return self.get_text(x)

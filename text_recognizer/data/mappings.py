"""Mapping to and from word pieces."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Sequence

from loguru import logger
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


class EmnistMapping(AbstractMapping):
    def __init__(self, extra_symbols: Optional[Sequence[str]]) -> None:
        self.mapping, self.inverse_mapping, self.input_size = emnist_mapping(
            extra_symbols
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


class WordPieceMapping(EmnistMapping):
    def __init__(
        self,
        num_features: int = 1000,
        tokens: str = "iamdb_1kwp_tokens_1000.txt" ,
        lexicon: str = "iamdb_1kwp_lex_1000.txt",
        data_dir: Optional[Union[str, Path]] = None,
        use_words: bool = False,
        prepend_wordsep: bool = False,
        special_tokens: Sequence[str] = ("<s>", "<e>", "<p>"),
        extra_symbols: Optional[Sequence[str]] = ("\n", ),
    ) -> None:
        super().__init__(extra_symbols)
        self.wordpiece_processor = self._configure_wordpiece_processor(
            num_features,
            tokens,
            lexicon,
            data_dir,
            use_words,
            prepend_wordsep,
            special_tokens,
            extra_symbols,
        )

    @staticmethod
    def _configure_wordpiece_processor(
        num_features: int,
        tokens: str,
        lexicon: str,
        data_dir: Optional[Union[str, Path]],
        use_words: bool,
        prepend_wordsep: bool,
        special_tokens: Optional[Sequence[str]],
        extra_symbols: Optional[Sequence[str]],
    ) -> Preprocessor:
        data_dir = (
            (Path(__file__).resolve().parents[2] / "data" / "downloaded" / "iam" / "iamdb")
            if data_dir is None
            else Path(data_dir)
        )

        logger.debug(f"Using data dir: {data_dir}")
        if not data_dir.exists():
            raise RuntimeError(f"Could not locate iamdb directory at {data_dir}")

        processed_path = (
            Path(__file__).resolve().parents[2] / "data" / "processed" / "iam_lines"
        )

        tokens_path = processed_path / tokens
        lexicon_path = processed_path / lexicon

        if extra_symbols is not None:
            special_tokens += extra_symbols

        return Preprocessor(
            data_dir,
            num_features,
            tokens_path,
            lexicon_path,
            use_words,
            prepend_wordsep,
            special_tokens,
        )

    def get_token(self, index: Union[int, Tensor]) -> str:
        if (index := int(index)) <= self.wordpiece_processor.num_tokens:
            return self.wordpiece_processor.tokens[index]
        raise KeyError(f"Index ({index}) not in mapping.")

    def get_index(self, token: str) -> Tensor:
        if token in self.wordpiece_processor.tokens:
            return torch.LongTensor(self.wordpiece_processor.tokens_to_index[token])
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

"""Word piece mapping."""
from pathlib import Path
from typing import List, Optional, Union, Set

import torch
from loguru import logger as log
from torch import Tensor

from text_recognizer.data.emnist_mapping import EmnistMapping
from text_recognizer.data.iam_preprocessor import Preprocessor


class WordPieceMapping(EmnistMapping):
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        num_features: int = 1000,
        tokens: str = "iamdb_1kwp_tokens_1000.txt",
        lexicon: str = "iamdb_1kwp_lex_1000.txt",
        use_words: bool = False,
        prepend_wordsep: bool = False,
        special_tokens: Set[str] = {"<s>", "<e>", "<p>"},
        extra_symbols: Set[str] = {"\n",},
    ) -> None:
        super().__init__(extra_symbols=extra_symbols)
        self.data_dir = (
            (
                Path(__file__).resolve().parents[2]
                / "data"
                / "downloaded"
                / "iam"
                / "iamdb"
            )
            if data_dir is None
            else Path(data_dir)
        )
        log.debug(f"Using data dir: {self.data_dir}")
        if not self.data_dir.exists():
            raise RuntimeError(f"Could not locate iamdb directory at {self.data_dir}")

        processed_path = (
            Path(__file__).resolve().parents[2] / "data" / "processed" / "iam_lines"
        )

        tokens_path = processed_path / tokens
        lexicon_path = processed_path / lexicon

        special_tokens = set(special_tokens)
        if self.extra_symbols is not None:
            special_tokens = special_tokens | set(extra_symbols)

        self.wordpiece_processor = Preprocessor(
            data_dir=self.data_dir,
            num_features=num_features,
            tokens_path=tokens_path,
            lexicon_path=lexicon_path,
            use_words=use_words,
            prepend_wordsep=prepend_wordsep,
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

    def __getitem__(self, x: Union[int, Tensor]) -> str:
        return self.get_token(x)

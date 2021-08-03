"""Transforms for PyTorch datasets."""
from pathlib import Path
from typing import Optional, Union, Set

import torch
from torch import Tensor

from text_recognizer.data.word_piece_mapping import WordPieceMapping


class WordPiece:
    """Converts EMNIST indices to Word Piece indices."""

    def __init__(
        self,
        num_features: int = 1000,
        tokens: str = "iamdb_1kwp_tokens_1000.txt",
        lexicon: str = "iamdb_1kwp_lex_1000.txt",
        data_dir: Optional[Union[str, Path]] = None,
        use_words: bool = False,
        prepend_wordsep: bool = False,
        special_tokens: Set[str] = {"<s>", "<e>", "<p>"},
        extra_symbols: Optional[Set[str]] = {"\n",},
        max_len: int = 451,
    ) -> None:
        self.mapping = WordPieceMapping(
            data_dir=data_dir,
            num_features=num_features,
            tokens=tokens,
            lexicon=lexicon,
            use_words=use_words,
            prepend_wordsep=prepend_wordsep,
            special_tokens=special_tokens,
            extra_symbols=extra_symbols,
        )
        self.max_len = max_len

    def __call__(self, x: Tensor) -> Tensor:
        y = self.mapping.emnist_to_wordpiece_indices(x)
        if len(y) < self.max_len:
            pad_len = self.max_len - len(y)
            y = torch.cat(
                (y, torch.LongTensor([self.mapping.get_index("<p>")] * pad_len))
            )
        else:
            y = y[: self.max_len]
        return y

"""Transforms for PyTorch datasets."""
from pathlib import Path
from typing import Optional, Union, Sequence

from torch import Tensor

from text_recognizer.data.mappings import WordPieceMapping


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
        special_tokens: Sequence[str] = ("<s>", "<e>", "<p>"),
        extra_symbols: Optional[Sequence[str]] = ("\n",),
        max_len: int = 192,
    ) -> None:
        self.mapping = WordPieceMapping(
            num_features,
            tokens,
            lexicon,
            data_dir,
            use_words,
            prepend_wordsep,
            special_tokens,
            extra_symbols,
        )
        self.max_len = max_len

    def __call__(self, x: Tensor) -> Tensor:
        return self.mapping.emnist_to_wordpiece_indices(x)[: self.max_len]

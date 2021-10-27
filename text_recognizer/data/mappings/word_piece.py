"""Word piece mapping."""
from typing import List, Set, Union

import torch
from torch import Tensor

from text_recognizer.data.mappings.emnist import EmnistMapping
from text_recognizer.data.utils.iam_preprocessor import Preprocessor


class WordPieceMapping(EmnistMapping):
    """Word piece mapping."""

    def __init__(
        self,
        num_features: int = 1000,
        tokens: str = "iamdb_1kwp_tokens_1000.txt",
        lexicon: str = "iamdb_1kwp_lex_1000.txt",
        use_words: bool = False,
        prepend_wordsep: bool = False,
        special_tokens: Set[str] = {"<s>", "<e>", "<p>"},
        extra_symbols: Set[str] = {"\n"},
    ) -> None:
        super().__init__(extra_symbols=extra_symbols)
        special_tokens = set(special_tokens)
        if self.extra_symbols is not None:
            special_tokens = special_tokens | set(extra_symbols)

        self.wordpiece_processor = Preprocessor(
            num_features=num_features,
            tokens=tokens,
            lexicon=lexicon,
            use_words=use_words,
            prepend_wordsep=prepend_wordsep,
            special_tokens=special_tokens,
        )

    def __len__(self) -> int:
        """Return number of word pieces."""
        return len(self.wordpiece_processor.tokens)

    def get_token(self, index: Union[int, Tensor]) -> str:
        """Returns token for index."""
        if (index := int(index)) <= self.wordpiece_processor.num_tokens:
            return self.wordpiece_processor.tokens[index]
        raise KeyError(f"Index ({index}) not in mapping.")

    def get_index(self, token: str) -> Tensor:
        """Returns index of token."""
        if token in self.wordpiece_processor.tokens:
            return torch.LongTensor([self.wordpiece_processor.tokens_to_index[token]])
        raise KeyError(f"Token ({token}) not found in inverse mapping.")

    def get_text(self, indices: Union[List[int], Tensor]) -> str:
        """Returns text from indices."""
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        return self.wordpiece_processor.to_text(indices)

    def get_indices(self, text: str) -> Tensor:
        """Returns indices of text."""
        return self.wordpiece_processor.to_index(text)

    def emnist_to_wordpiece_indices(self, x: Tensor) -> Tensor:
        """Returns word pieces indices from emnist indices."""
        text = "".join([self.mapping[i] for i in x])
        text = text.lower().replace(" ", "▁")
        return torch.LongTensor(self.wordpiece_processor.to_index(text))

    def __getitem__(self, x: Union[int, Tensor]) -> str:
        """Returns token for word piece index."""
        return self.get_token(x)

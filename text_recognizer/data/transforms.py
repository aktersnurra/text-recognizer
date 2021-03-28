"""Transforms for PyTorch datasets."""
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger
import torch
from torch import Tensor

from text_recognizer.datasets.iam_preprocessor import Preprocessor
from text_recognizer.data.emnist import emnist_mapping
       
class ToLower:
    """Converts target to lower case."""

    def __call__(self, target: Tensor) -> Tensor:
        """Corrects index value in target tensor."""
        device = target.device
        return torch.stack([x - 26 if x > 35 else x for x in target]).to(device)


class ToCharcters:
    """Converts integers to characters."""

    def __init__(self) -> None:
        self.mapping, _, _ = emnist_mapping() 

    def __call__(self, y: Tensor) -> str:
        """Converts a Tensor to a str."""
        return (
            "".join([self.mapping(int(i)) for i in y])
            .strip("<p>")
            .replace(" ", "▁")
        )


class WordPieces:
    """Abstract transform for word pieces."""

    def __init__(
        self,
        num_features: int,
        data_dir: Optional[Union[str, Path]] = None,
        tokens: Optional[Union[str, Path]] = None,
        lexicon: Optional[Union[str, Path]] = None,
        use_words: bool = False,
        prepend_wordsep: bool = False,
    ) -> None:
        if data_dir is None:
            data_dir = (
                Path(__file__).resolve().parents[3] / "data" / "raw" / "iam" / "iamdb"
            )
            logger.debug(f"Using data dir: {data_dir}")
            if not data_dir.exists():
                raise RuntimeError(f"Could not locate iamdb directory at {data_dir}")
        else:
            data_dir = Path(data_dir)
        processed_path = (
            Path(__file__).resolve().parents[3] / "data" / "processed" / "iam_lines"
        )
        tokens_path = processed_path / tokens
        lexicon_path = processed_path / lexicon

        self.preprocessor = Preprocessor(
            data_dir,
            num_features,
            tokens_path,
            lexicon_path,
            use_words,
            prepend_wordsep,
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Transforms input."""
        ...


class ToWordPieces(WordPieces):
    """Transforms str to word pieces."""

    def __init__(
        self,
        num_features: int,
        data_dir: Optional[Union[str, Path]] = None,
        tokens: Optional[Union[str, Path]] = None,
        lexicon: Optional[Union[str, Path]] = None,
        use_words: bool = False,
        prepend_wordsep: bool = False,
    ) -> None:
        super().__init__(
            num_features, data_dir, tokens, lexicon, use_words, prepend_wordsep
        )

    def __call__(self, line: str) -> Tensor:
        """Transforms str to word pieces."""
        return self.preprocessor.to_index(line)


class ToText(WordPieces):
    """Takes word pieces and converts them to text."""

    def __init__(
        self,
        num_features: int,
        data_dir: Optional[Union[str, Path]] = None,
        tokens: Optional[Union[str, Path]] = None,
        lexicon: Optional[Union[str, Path]] = None,
        use_words: bool = False,
        prepend_wordsep: bool = False,
    ) -> None:
        super().__init__(
            num_features, data_dir, tokens, lexicon, use_words, prepend_wordsep
        )

    def __call__(self, x: Tensor) -> str:
        """Converts tensor to text."""
        return self.preprocessor.to_text(x.tolist())

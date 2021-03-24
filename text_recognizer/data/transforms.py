"""Transforms for PyTorch datasets."""
from abc import abstractmethod
from pathlib import Path
import random
from typing import Any, Optional, Union

from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomAffine,
    RandomHorizontalFlip,
    RandomRotation,
    ToPILImage,
    ToTensor,
)

from text_recognizer.datasets.iam_preprocessor import Preprocessor
from text_recognizer.datasets.util import EmnistMapper


class RandomResizeCrop:
    """Image transform with random resize and crop applied.

    Stolen from

    https://github.com/facebookresearch/gtn_applications/blob/master/datasets/iamdb.py

    """

    def __init__(self, jitter: int = 10, ratio: float = 0.5) -> None:
        self.jitter = jitter
        self.ratio = ratio

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Applies random crop and rotation to an image."""
        w, h = img.size

        # pad with white:
        img = transforms.functional.pad(img, self.jitter, fill=255)

        # crop at random (x, y):
        x = self.jitter + random.randint(-self.jitter, self.jitter)
        y = self.jitter + random.randint(-self.jitter, self.jitter)

        # randomize aspect ratio:
        size_w = w * random.uniform(1 - self.ratio, 1 + self.ratio)
        size = (h, int(size_w))
        img = transforms.functional.resized_crop(img, y, x, h, w, size)
        return img


class Transpose:
    """Transposes the EMNIST image to the correct orientation."""

    def __call__(self, image: Image) -> np.ndarray:
        """Swaps axis."""
        return np.array(image).swapaxes(0, 1)


class Resize:
    """Resizes a tensor to a specified width."""

    def __init__(self, width: int = 952) -> None:
        # The default is 952 because of the IAM dataset.
        self.width = width

    def __call__(self, image: Tensor) -> Tensor:
        """Resize tensor in the last dimension."""
        return F.interpolate(image, size=self.width, mode="nearest")


class AddTokens:
    """Adds start of sequence and end of sequence tokens to target tensor."""

    def __init__(self, pad_token: str, eos_token: str, init_token: str = None) -> None:
        self.init_token = init_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        if self.init_token is not None:
            self.emnist_mapper = EmnistMapper(
                init_token=self.init_token,
                pad_token=self.pad_token,
                eos_token=self.eos_token,
            )
        else:
            self.emnist_mapper = EmnistMapper(
                pad_token=self.pad_token, eos_token=self.eos_token,
            )
        self.pad_value = self.emnist_mapper(self.pad_token)
        self.eos_value = self.emnist_mapper(self.eos_token)

    def __call__(self, target: Tensor) -> Tensor:
        """Adds a sos token to the begining and a eos token to the end of a target sequence."""
        dtype, device = target.dtype, target.device

        # Find the where padding starts.
        pad_index = torch.nonzero(target == self.pad_value, as_tuple=False)[0].item()

        target[pad_index] = self.eos_value

        if self.init_token is not None:
            self.sos_value = self.emnist_mapper(self.init_token)
            sos = torch.tensor([self.sos_value], dtype=dtype, device=device)
            target = torch.cat([sos, target], dim=0)

        return target


class ApplyContrast:
    """Sets everything below a threshold to zero, i.e. increase contrast."""

    def __init__(self, low: float = 0.0, high: float = 0.25) -> None:
        self.low = low
        self.high = high

    def __call__(self, x: Tensor) -> Tensor:
        """Apply mask binary mask to input tensor."""
        mask = x > np.random.RandomState().uniform(low=self.low, high=self.high)
        return x * mask


class Unsqueeze:
    """Add a dimension to the tensor."""

    def __call__(self, x: Tensor) -> Tensor:
        """Adds dim."""
        return x.unsqueeze(0)


class Squeeze:
    """Removes the first dimension of a tensor."""

    def __call__(self, x: Tensor) -> Tensor:
        """Removes first dim."""
        return x.squeeze(0)


class ToLower:
    """Converts target to lower case."""

    def __call__(self, target: Tensor) -> Tensor:
        """Corrects index value in target tensor."""
        device = target.device
        return torch.stack([x - 26 if x > 35 else x for x in target]).to(device)


class ToCharcters:
    """Converts integers to characters."""

    def __init__(
        self, pad_token: str, eos_token: str, init_token: str = None, lower: bool = True
    ) -> None:
        self.init_token = init_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        if self.init_token is not None:
            self.emnist_mapper = EmnistMapper(
                init_token=self.init_token,
                pad_token=self.pad_token,
                eos_token=self.eos_token,
                lower=lower,
            )
        else:
            self.emnist_mapper = EmnistMapper(
                pad_token=self.pad_token, eos_token=self.eos_token, lower=lower
            )

    def __call__(self, y: Tensor) -> str:
        """Converts a Tensor to a str."""
        return (
            "".join([self.emnist_mapper(int(i)) for i in y])
            .strip("_")
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

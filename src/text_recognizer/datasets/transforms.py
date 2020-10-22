"""Transforms for PyTorch datasets."""
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import Compose, ToTensor

from text_recognizer.datasets.util import EmnistMapper


class Transpose:
    """Transposes the EMNIST image to the correct orientation."""

    def __call__(self, image: Image) -> np.ndarray:
        """Swaps axis."""
        return np.array(image).swapaxes(0, 1)


class AddTokens:
    """Adds start of sequence and end of sequence tokens to target tensor."""

    def __init__(self, init_token: str, pad_token: str, eos_token: str,) -> None:
        self.init_token = init_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.emnist_mapper = EmnistMapper(
            init_token=self.init_token,
            pad_token=self.pad_token,
            eos_token=self.eos_token,
        )
        self.pad_value = self.emnist_mapper(self.pad_token)
        self.sos_value = self.emnist_mapper(self.init_token)
        self.eos_value = self.emnist_mapper(self.eos_token)

    def __call__(self, target: Tensor) -> Tensor:
        """Adds a sos token to the begining and a eos token to the end of a target sequence."""
        dtype, device = target.dtype, target.device
        sos = torch.tensor([self.sos_value], dtype=dtype, device=device)

        # Find the where padding starts.
        pad_index = torch.nonzero(target == self.pad_value, as_tuple=False)[0].item()

        target[pad_index] = self.eos_value

        target = torch.cat([sos, target], dim=0)
        return target

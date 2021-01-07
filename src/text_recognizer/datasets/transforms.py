"""Transforms for PyTorch datasets."""
import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms import Compose, RandomAffine, RandomHorizontalFlip, ToTensor

from text_recognizer.datasets.util import EmnistMapper


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

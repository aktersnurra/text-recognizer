"""Transforms for PyTorch datasets."""
import numpy as np
from PIL import Image
import torch
from torch import Tensor


class Transpose:
    """Transposes the EMNIST image to the correct orientation."""

    def __call__(self, image: Image) -> np.ndarray:
        """Swaps axis."""
        return np.array(image).swapaxes(0, 1)

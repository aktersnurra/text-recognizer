"""Util functions for datasets."""
import numpy as np
from PIL import Image


class Transpose:
    """Transposes the EMNIST image to the correct orientation."""

    def __call__(self, image: Image) -> np.ndarray:
        """Swaps axis."""
        return np.array(image).swapaxes(0, 1)

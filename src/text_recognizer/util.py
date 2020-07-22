"""Utility functions for text_recognizer module."""
import os
from pathlib import Path
from typing import Union
from urllib.request import urlopen

import cv2
import numpy as np


def read_image(image_uri: Union[Path, str], grayscale: bool = False) -> np.ndarray:
    """Read image_uri."""

    def read_image_from_filename(image_filename: str, imread_flag: int) -> np.ndarray:
        return cv2.imread(str(image_filename), imread_flag)

    def read_image_from_url(image_url: str, imread_flag: int) -> np.ndarray:
        if image_url.lower().startswith("http"):
            url_response = urlopen(str(image_url))
            image_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
            return cv2.imdecode(image_array, imread_flag)
        else:
            raise ValueError(
                "Url does not start with http, therfore not safe to open..."
            ) from None

    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    local_file = os.path.exists(image_uri)
    try:
        image = None
        if local_file:
            image = read_image_from_filename(image_uri, imread_flag)
        else:
            image = read_image_from_url(image_uri, imread_flag)
        assert image is not None
    except Exception as e:
        raise ValueError(f"Could not load image at {image_uri}: {e}")
    return image


def rescale_image(image: np.ndarray) -> np.ndarray:
    """Rescale image from [0, 1] to [0, 255]."""
    if image.max() <= 1.0:
        image = 255 * (image - image.min()) / (image.max() - image.min())
    return image


def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
    """Write image to file."""
    image = rescale_image(image)
    cv2.imwrite(str(filename), image)

"""Image util functions for loading and saving images."""
from pathlib import Path
from typing import Union
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image


def read_image_pil(image_uri: Union[Path, str], grayscale: bool = False) -> Image:
    """Return PIL image."""
    image = Image.open(image_uri)
    if grayscale:
        image = image.convert("L")
    return image


def read_image(image_uri: Union[Path, str], grayscale: bool = False) -> np.array:
    """Read image_uri."""

    if isinstance(image_uri, str):
        image_uri = Path(image_uri)

    def read_image_from_filename(image_filename: Path, imread_flag: int) -> np.array:
        return cv2.imread(str(image_filename), imread_flag)

    def read_image_from_url(image_url: Path, imread_flag: int) -> np.array:
        url_response = urlopen(str(image_url))  # nosec
        image_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        return cv2.imdecode(image_array, imread_flag)

    imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = None

    if image_uri.exists():
        image = read_image_from_filename(image_uri, imread_flag)
    else:
        image = read_image_from_url(image_uri, imread_flag)

    if image is None:
        raise ValueError(f"Could not load image at {image_uri}")

    return image


def write_image(image: np.ndarray, filename: Union[Path, str]) -> None:
    """Write image to file."""
    cv2.imwrite(str(filename), image)

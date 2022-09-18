"""Image util functions for loading and saving images."""
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import smart_open
from PIL import Image


def read_image_pil(image_uri: Union[Path, str], grayscale: bool = False) -> Image:
    """Read image from uri."""
    with smart_open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file: str, grayscale: bool = False) -> Image:
    """Return PIL image."""
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert("L")
        else:
            image = image.convert(mode=image.mode)
    return image


def read_b64_image(b64_str: str, grayscale: bool = False) -> Image:
    """Load base64-encoded images."""
    try:
        _, b64_data = b64_str.split(",")
        image_file = BytesIO(base64.b64decode(b64_data))
        return read_image_pil_file(image_file, grayscale)
    except Exception as e:
        raise ValueError(f"Could not load image from b64 {b64_str}: {e}")

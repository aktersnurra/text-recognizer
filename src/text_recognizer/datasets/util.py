"""Util functions for datasets."""
import hashlib
import importlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union
from urllib.request import urlopen, urlretrieve

import cv2
from loguru import logger
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Transpose:
    """Transposes the EMNIST image to the correct orientation."""

    def __call__(self, image: Image) -> np.ndarray:
        """Swaps axis."""
        return np.array(image).swapaxes(0, 1)


def compute_sha256(filename: Union[Path, str]) -> str:
    """Returns the SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """TQDM progress bar when downloading files.

    From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py

    """

    def update_to(
        self, blocks: int = 1, block_size: int = 1, total_size: Optional[int] = None
    ) -> None:
        """Updates the progress bar.

        Args:
            blocks (int): Number of blocks transferred so far. Defaults to 1.
            block_size (int): Size of each block, in tqdm units. Defaults to 1.
            total_size (Optional[int]): Total size in tqdm units. Defaults to None.
        """
        if total_size is not None:
            self.total = total_size  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * block_size - self.n)


def download_url(url: str, filename: str) -> None:
    """Downloads a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


def _download_raw_dataset(metadata: Dict) -> None:
    if os.path.exists(metadata["filename"]):
        return
    logger.info(f"Downloading raw dataset from {metadata['url']}...")
    download_url(metadata["url"], metadata["filename"])
    logger.info("Computing SHA-256...")
    sha256 = compute_sha256(metadata["filename"])
    if sha256 != metadata["sha256"]:
        raise ValueError(
            "Downloaded data file SHA-256 does not match that listed in metadata document."
        )

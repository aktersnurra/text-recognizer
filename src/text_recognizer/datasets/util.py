"""Util functions for datasets."""
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union
from urllib.request import urlopen, urlretrieve

import cv2
from loguru import logger
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import EMNIST
from tqdm import tqdm

DATA_DIRNAME = Path(__file__).resolve().parents[3] / "data"
ESSENTIALS_FILENAME = Path(__file__).resolve().parents[0] / "emnist_essentials.json"


def save_emnist_essentials(emnsit_dataset: type = EMNIST) -> None:
    """Extract and saves EMNIST essentials."""
    labels = emnsit_dataset.classes
    labels.sort()
    mapping = [(i, str(label)) for i, label in enumerate(labels)]
    essentials = {
        "mapping": mapping,
        "input_shape": tuple(np.array(emnsit_dataset[0][0]).shape[:]),
    }
    logger.info("Saving emnist essentials...")
    with open(ESSENTIALS_FILENAME, "w") as f:
        json.dump(essentials, f)


def download_emnist() -> None:
    """Download the EMNIST dataset via the PyTorch class."""
    logger.info(f"Data directory is: {DATA_DIRNAME}")
    dataset = EMNIST(root=DATA_DIRNAME, split="byclass", download=True)
    save_emnist_essentials(dataset)


class EmnistMapper:
    """Mapper between network output to Emnist character."""

    def __init__(self) -> None:
        """Loads the emnist essentials file with the mapping and input shape."""
        self.essentials = self._load_emnist_essentials()
        # Load dataset infromation.
        self._mapping = self._augment_emnist_mapping(dict(self.essentials["mapping"]))
        self._inverse_mapping = {v: k for k, v in self.mapping.items()}
        self._num_classes = len(self.mapping)
        self._input_shape = self.essentials["input_shape"]

    def __call__(self, token: Union[str, int, np.uint8]) -> Union[str, int]:
        """Maps the token to emnist character or character index.

        If the token is an integer (index), the method will return the Emnist character corresponding to that index.
        If the token is a str (Emnist character), the method will return the corresponding index for that character.

        Args:
            token (Union[str, int, np.uint8]): Eihter a string or index (integer).

        Returns:
            Union[str, int]: The mapping result.

        Raises:
            KeyError: If the index or string does not exist in the mapping.

        """
        if (isinstance(token, np.uint8) or isinstance(token, int)) and int(
            token
        ) in self.mapping:
            return self.mapping[int(token)]
        elif isinstance(token, str) and token in self._inverse_mapping:
            return self._inverse_mapping[token]
        else:
            raise KeyError(f"Token {token} does not exist in the mappings.")

    @property
    def mapping(self) -> Dict:
        """Returns the mapping between index and character."""
        return self._mapping

    @property
    def inverse_mapping(self) -> Dict:
        """Returns the mapping between character and index."""
        return self._inverse_mapping

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in the dataset."""
        return self._num_classes

    @property
    def input_shape(self) -> List[int]:
        """Returns the input shape of the Emnist characters."""
        return self._input_shape

    def _load_emnist_essentials(self) -> Dict:
        """Load the EMNIST mapping."""
        with open(str(ESSENTIALS_FILENAME)) as f:
            essentials = json.load(f)
        return essentials

    def _augment_emnist_mapping(self, mapping: Dict) -> Dict:
        """Augment the mapping with extra symbols."""
        # Extra symbols in IAM dataset
        extra_symbols = [
            " ",
            "!",
            '"',
            "#",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
            ":",
            ";",
            "?",
        ]

        # padding symbol, and acts as blank symbol as well.
        extra_symbols.append("_")

        max_key = max(mapping.keys())
        extra_mapping = {}
        for i, symbol in enumerate(extra_symbols):
            extra_mapping[max_key + 1 + i] = symbol

        return {**mapping, **extra_mapping}


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

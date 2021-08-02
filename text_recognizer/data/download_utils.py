"""Util functions for downloading datasets."""
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve

from loguru import logger as log
from tqdm import tqdm


def _compute_sha256(filename: Path) -> str:
    """Returns the SHA256 checksum of a file."""
    with filename.open(mode="rb") as f:
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
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download_url(url: str, filename: str) -> None:
    """Downloads a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


def download_dataset(metadata: Dict, dl_dir: Path) -> Optional[Path]:
    """Downloads dataset using a metadata file.

    Args:
        metadata (Dict): A metadata file of the dataset.
        dl_dir (Path): Download directory for the dataset.

    Returns:
        Optional[Path]: Returns filename if dataset is downloaded, None if it already
            exists.

    Raises:
        ValueError: If the SHA-256 value is not the same between the dataset and
            the metadata file.

    """
    dl_dir.mkdir(parents=True, exist_ok=True)
    filename = dl_dir / metadata["filename"]
    if filename.exists():
        return
    log.info(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    _download_url(metadata["url"], filename)
    log.info("Computing the SHA-256...")
    sha256 = _compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError(
            "Downloaded data file SHA-256 does not match that listed in metadata document."
        )
    return filename

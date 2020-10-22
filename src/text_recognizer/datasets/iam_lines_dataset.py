"""IamLinesDataset class."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
from loguru import logger
import torch
from torch import Tensor
from torchvision.transforms import ToTensor

from text_recognizer.datasets.dataset import Dataset
from text_recognizer.datasets.util import (
    compute_sha256,
    DATA_DIRNAME,
    download_url,
    EmnistMapper,
)


PROCESSED_DATA_DIRNAME = DATA_DIRNAME / "processed" / "iam_lines"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "iam_lines.h5"
PROCESSED_DATA_URL = (
    "https://s3-us-west-2.amazonaws.com/fsdl-public-assets/iam_lines.h5"
)


class IamLinesDataset(Dataset):
    """IAM lines datasets for handwritten text lines."""

    def __init__(
        self,
        train: bool = False,
        subsample_fraction: float = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        init_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        eos_token: Optional[str] = None,
    ) -> None:
        super().__init__(
            train=train,
            subsample_fraction=subsample_fraction,
            transform=transform,
            target_transform=target_transform,
            init_token=init_token,
            pad_token=pad_token,
            eos_token=eos_token,
        )

    @property
    def input_shape(self) -> Tuple:
        """Input shape of the data."""
        return self.data.shape[1:] if self.data is not None else None

    @property
    def output_shape(self) -> Tuple:
        """Output shape of the data."""
        return (
            self.targets.shape[1:] + (self.num_classes,)
            if self.targets is not None
            else None
        )

    def load_or_generate_data(self) -> None:
        """Load or generate dataset data."""
        if not PROCESSED_DATA_FILENAME.exists():
            PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading IAM lines...")
            download_url(PROCESSED_DATA_URL, PROCESSED_DATA_FILENAME)
        with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
            self._data = f[f"x_{self.split}"][:]
            self._targets = f[f"y_{self.split}"][:]
        self._subsample()

    def __repr__(self) -> str:
        """Print info about the dataset."""
        return (
            "IAM Lines Dataset\n"  # pylint: disable=no-member
            f"Number classes: {self.num_classes}\n"
            f"Mapping: {self.mapper.mapping}\n"
            f"Data: {self.data.shape}\n"
            f"Targets: {self.targets.shape}\n"
        )

    def __getitem__(self, index: Union[Tensor, int]) -> Tuple[Tensor, Tensor]:
        """Fetches data, target pair of the dataset for a given and index or indices.

        Args:
            index (Union[int, Tensor]): Either a list or int of indices/index.

        Returns:
            Tuple[Tensor, Tensor]: Data target pair.

        """
        if torch.is_tensor(index):
            index = index.tolist()

        data = self.data[index]
        targets = self.targets[index]

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            targets = self.target_transform(targets)

        return data, targets

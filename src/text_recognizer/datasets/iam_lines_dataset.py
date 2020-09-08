"""IamLinesDataset class."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
from loguru import logger
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from text_recognizer.datasets.emnist_dataset import DATA_DIRNAME, EmnistMapper
from text_recognizer.datasets.util import compute_sha256, download_url


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
    ) -> None:
        self.train = train
        self.split = "train" if self.train else "test"
        self._mapper = EmnistMapper()
        self.num_classes = self.mapper.num_classes

        # Set transforms.
        self.transform = transform
        if self.transform is None:
            self.transform = ToTensor()

        self.target_transform = target_transform
        if self.target_transform is None:
            self.target_transform = torch.tensor

        self.subsample_fraction = subsample_fraction
        self.data = None
        self.targets = None

    @property
    def mapper(self) -> EmnistMapper:
        """Returns the EmnistMapper."""
        return self._mapper

    @property
    def mapping(self) -> Dict:
        """Return EMNIST mapping from index to character."""
        return self._mapper.mapping

    @property
    def input_shape(self) -> Tuple:
        """Input shape of the data."""
        return self.data.shape[1:]

    @property
    def output_shape(self) -> Tuple:
        """Output shape of the data."""
        return self.targets.shape[1:] + (self.num_classes,)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def load_or_generate_data(self) -> None:
        """Load or generate dataset data."""
        if not PROCESSED_DATA_FILENAME.exists():
            PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading IAM lines...")
            download_url(PROCESSED_DATA_URL, PROCESSED_DATA_FILENAME)
        with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
            self.data = f[f"x_{self.split}"][:]
            self.targets = f[f"y_{self.split}"][:]
        self._subsample()

    def _subsample(self) -> None:
        """Only a fraction of the data will be loaded."""
        if self.subsample_fraction is None:
            return

        num_samples = int(self.data.shape[0] * self.subsample_fraction)
        self.data = self.data[:num_samples]
        self.targets = self.targets[:num_samples]

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

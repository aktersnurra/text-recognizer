"""Abstract dataset class."""
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils import data
from torchvision.transforms import ToTensor

from text_recognizer.datasets.util import EmnistMapper


class Dataset(data.Dataset):
    """Abstract class for with common methods for all datasets."""

    def __init__(
        self,
        train: bool,
        subsample_fraction: float = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        init_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        eos_token: Optional[str] = None,
    ) -> None:
        """Initialization of Dataset class.

        Args:
            train (bool): If True, loads the training set, otherwise the validation set is loaded. Defaults to False.
            subsample_fraction (float): The fraction of the dataset to use for training. Defaults to None.
            transform (Optional[Callable]): Transform(s) for input data. Defaults to None.
            target_transform (Optional[Callable]): Transform(s) for output data. Defaults to None.
            init_token (Optional[str]): String representing the start of sequence token. Defaults to None.
            pad_token (Optional[str]): String representing the pad token. Defaults to None.
            eos_token (Optional[str]): String representing the end of sequence token. Defaults to None.

        Raises:
            ValueError: If subsample_fraction is not None and outside the range (0, 1).

        """
        self.train = train
        self.split = "train" if self.train else "test"

        if subsample_fraction is not None:
            if not 0.0 < subsample_fraction < 1.0:
                raise ValueError("The subsample fraction must be in (0, 1).")
        self.subsample_fraction = subsample_fraction

        self._mapper = EmnistMapper(
            init_token=init_token, eos_token=eos_token, pad_token=pad_token
        )
        self._input_shape = self._mapper.input_shape
        self._output_shape = self._mapper._num_classes
        self.num_classes = self.mapper.num_classes

        # Set transforms.
        self.transform = transform if transform is not None else ToTensor()
        self.target_transform = (
            target_transform if target_transform is not None else torch.tensor
        )

        self._data = None
        self._targets = None

    @property
    def data(self) -> Tensor:
        """The input data."""
        return self._data

    @property
    def targets(self) -> Tensor:
        """The target data."""
        return self._targets

    @property
    def input_shape(self) -> Tuple:
        """Input shape of the data."""
        return self._input_shape

    @property
    def output_shape(self) -> Tuple:
        """Output shape of the data."""
        return self._output_shape

    @property
    def mapper(self) -> EmnistMapper:
        """Returns the EmnistMapper."""
        return self._mapper

    @property
    def mapping(self) -> Dict:
        """Return EMNIST mapping from index to character."""
        return self._mapper.mapping

    @property
    def inverse_mapping(self) -> Dict:
        """Returns the inverse mapping from character to index."""
        return self.mapper.inverse_mapping

    def _subsample(self) -> None:
        """Only this fraction of the data will be loaded."""
        if self.subsample_fraction is None:
            return
        num_subsample = int(self.data.shape[0] * self.subsample_fraction)
        self._data = self.data[:num_subsample]
        self._targets = self.targets[:num_subsample]

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def load_or_generate_data(self) -> None:
        """Load or generate dataset data."""
        raise NotImplementedError

    def __getitem__(self, index: Union[int, Tensor]) -> Tuple[Tensor, Tensor]:
        """Fetches samples from the dataset.

        Args:
            index (Union[int, torch.Tensor]): The indices of the samples to fetch.

        Raises:
            NotImplementedError: If the method is not implemented in child class.

        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Returns information about the dataset."""
        raise NotImplementedError

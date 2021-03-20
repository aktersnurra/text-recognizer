"""Emnist dataset: black and white images of handwritten characters (Aa-Zz) and digits (0-9)."""

import json
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor

from text_recognizer.datasets.dataset import Dataset
from text_recognizer.datasets.transforms import Transpose
from text_recognizer.datasets.util import DATA_DIRNAME


class EmnistDataset(Dataset):
    """This is a class for resampling and subsampling the PyTorch EMNIST dataset."""

    def __init__(
        self,
        pad_token: str = None,
        train: bool = False,
        sample_to_balance: bool = False,
        subsample_fraction: float = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 4711,
    ) -> None:
        """Loads the dataset and the mappings.

        Args:
            pad_token (str): The pad token symbol. Defaults to _.
            train (bool): If True, loads the training set, otherwise the validation set is loaded. Defaults to False.
            sample_to_balance (bool): Resamples the dataset to make it balanced. Defaults to False.
            subsample_fraction (float): Description of parameter `subsample_fraction`. Defaults to None.
            transform (Optional[Callable]): Transform(s) for input data. Defaults to None.
            target_transform (Optional[Callable]): Transform(s) for output data. Defaults to None.
            seed (int): Seed number. Defaults to 4711.

        """
        super().__init__(
            train=train,
            subsample_fraction=subsample_fraction,
            transform=transform,
            target_transform=target_transform,
            pad_token=pad_token,
        )

        self.sample_to_balance = sample_to_balance

        # Have to transpose the emnist characters, ToTensor norms input between [0,1].
        if transform is None:
            self.transform = Compose([Transpose(), ToTensor()])

        self.target_transform = None

        self.seed = seed

    def __getitem__(self, index: Union[int, Tensor]) -> Tuple[Tensor, Tensor]:
        """Fetches samples from the dataset.

        Args:
            index (Union[int, Tensor]): The indices of the samples to fetch.

        Returns:
            Tuple[Tensor, Tensor]: Data target tuple.

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

    def __repr__(self) -> str:
        """Returns information about the dataset."""
        return (
            "EMNIST Dataset\n"
            f"Num classes: {self.num_classes}\n"
            f"Input shape: {self.input_shape}\n"
            f"Mapping: {self.mapper.mapping}\n"
        )

    def _sample_to_balance(self) -> None:
        """Because the dataset is not balanced, we take at most the mean number of instances per class."""
        np.random.seed(self.seed)
        x = self._data
        y = self._targets
        num_to_sample = int(np.bincount(y.flatten()).mean())
        all_sampled_indices = []
        for label in np.unique(y.flatten()):
            inds = np.where(y == label)[0]
            sampled_indices = np.unique(np.random.choice(inds, num_to_sample))
            all_sampled_indices.append(sampled_indices)
        indices = np.concatenate(all_sampled_indices)
        x_sampled = x[indices]
        y_sampled = y[indices]
        self._data = x_sampled
        self._targets = y_sampled

    def load_or_generate_data(self) -> None:
        """Fetch the EMNIST dataset."""
        dataset = EMNIST(
            root=DATA_DIRNAME,
            split="byclass",
            train=self.train,
            download=False,
            transform=None,
            target_transform=None,
        )

        self._data = dataset.data
        self._targets = dataset.targets

        if self.sample_to_balance:
            self._sample_to_balance()

        if self.subsample_fraction is not None:
            self._subsample()

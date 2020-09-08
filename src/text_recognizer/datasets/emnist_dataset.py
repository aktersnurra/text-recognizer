"""Emnist dataset: black and white images of handwritten characters (Aa-Zz) and digits (0-9)."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from text_recognizer.datasets.util import Transpose

DATA_DIRNAME = Path(__file__).resolve().parents[3] / "data"
ESSENTIALS_FILENAME = Path(__file__).resolve().parents[0] / "emnist_essentials.json"


def save_emnist_essentials(emnsit_dataset: type = EMNIST) -> None:
    """Extract and saves EMNIST essentials."""
    labels = emnsit_dataset.classes
    labels.sort()
    mapping = [(i, str(label)) for i, label in enumerate(labels)]
    essentials = {
        "mapping": mapping,
        "input_shape": tuple(emnsit_dataset[0][0].shape[:]),
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

        # padding symbol
        extra_symbols.append("_")

        max_key = max(mapping.keys())
        extra_mapping = {}
        for i, symbol in enumerate(extra_symbols):
            extra_mapping[max_key + 1 + i] = symbol

        return {**mapping, **extra_mapping}


class EmnistDataset(Dataset):
    """This is a class for resampling and subsampling the PyTorch EMNIST dataset."""

    def __init__(
        self,
        train: bool = False,
        sample_to_balance: bool = False,
        subsample_fraction: float = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 4711,
    ) -> None:
        """Loads the dataset and the mappings.

        Args:
            train (bool): If True, loads the training set, otherwise the validation set is loaded. Defaults to False.
            sample_to_balance (bool): Resamples the dataset to make it balanced. Defaults to False.
            subsample_fraction (float): Description of parameter `subsample_fraction`. Defaults to None.
            transform (Optional[Callable]): Transform(s) for input data. Defaults to None.
            target_transform (Optional[Callable]): Transform(s) for output data. Defaults to None.
            seed (int): Seed number. Defaults to 4711.

        Raises:
            ValueError: If subsample_fraction is not None and outside the range (0, 1).

        """

        self.train = train
        self.sample_to_balance = sample_to_balance

        if subsample_fraction is not None:
            if not 0.0 < subsample_fraction < 1.0:
                raise ValueError("The subsample fraction must be in (0, 1).")
        self.subsample_fraction = subsample_fraction

        self.transform = transform
        if self.transform is None:
            self.transform = Compose([Transpose(), ToTensor()])

        self.target_transform = target_transform
        self.seed = seed

        self._mapper = EmnistMapper()
        self._input_shape = self._mapper.input_shape
        self.num_classes = self._mapper.num_classes

        # Load dataset.
        self._data, self._targets = self.load_emnist_dataset()

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
    def mapper(self) -> EmnistMapper:
        """Returns the EmnistMapper."""
        return self._mapper

    @property
    def inverse_mapping(self) -> Dict:
        """Returns the inverse mapping from character to index."""
        return self.mapper.inverse_mapping

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: Union[int, Tensor]) -> Tuple[Tensor, Tensor]:
        """Fetches samples from the dataset.

        Args:
            index (Union[int, torch.Tensor]): The indices of the samples to fetch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Data target tuple.

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

    def _sample_to_balance(
        self, data: Tensor, targets: Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Because the dataset is not balanced, we take at most the mean number of instances per class."""
        np.random.seed(self.seed)
        x = data
        y = targets
        num_to_sample = int(np.bincount(y.flatten()).mean())
        all_sampled_indices = []
        for label in np.unique(y.flatten()):
            inds = np.where(y == label)[0]
            sampled_indices = np.unique(np.random.choice(inds, num_to_sample))
            all_sampled_indices.append(sampled_indices)
        indices = np.concatenate(all_sampled_indices)
        x_sampled = x[indices]
        y_sampled = y[indices]
        data = x_sampled
        targets = y_sampled
        return data, targets

    def _subsample(self, data: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Subsamples the dataset to the specified fraction."""
        x = data
        y = targets
        num_samples = int(x.shape[0] * self.subsample_fraction)
        x_sampled = x[:num_samples]
        y_sampled = y[:num_samples]
        self.data = x_sampled
        self.targets = y_sampled
        return data, targets

    def load_emnist_dataset(self) -> Tuple[Tensor, Tensor]:
        """Fetch the EMNIST dataset."""
        dataset = EMNIST(
            root=DATA_DIRNAME,
            split="byclass",
            train=self.train,
            download=False,
            transform=None,
            target_transform=None,
        )

        data = dataset.data
        targets = dataset.targets

        if self.sample_to_balance:
            data, targets = self._sample_to_balance(data, targets)

        if self.subsample_fraction is not None:
            data, targets = self._subsample(data, targets)

        return data, targets

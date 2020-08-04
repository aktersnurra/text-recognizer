"""Emnist dataset: black and white images of handwritten characters (Aa-Zz) and digits (0-9)."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from loguru import logger
import numpy as np
from PIL import Image
import torch
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


def _load_emnist_essentials() -> Dict:
    """Load the EMNIST mapping."""
    with open(str(ESSENTIALS_FILENAME)) as f:
        essentials = json.load(f)
    return essentials


def _augment_emnist_mapping(mapping: Dict) -> Dict:
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
            train (bool): If True, loads the training set, otherwise the validation set is loaded. Defaults to
                False.
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

        # Load dataset infromation.
        essentials = _load_emnist_essentials()
        self.mapping = _augment_emnist_mapping(dict(essentials["mapping"]))
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.num_classes = len(self.mapping)
        self.input_shape = essentials["input_shape"]

        # Placeholders
        self.data = None
        self.targets = None

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(
        self, index: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            f"Mapping: {self.mapping}\n"
            f"Input shape: {self.input_shape}\n"
        )

    def _sample_to_balance(self) -> None:
        """Because the dataset is not balanced, we take at most the mean number of instances per class."""
        np.random.seed(self.seed)
        x = self.data
        y = self.targets
        num_to_sample = int(np.bincount(y.flatten()).mean())
        all_sampled_indices = []
        for label in np.unique(y.flatten()):
            inds = np.where(y == label)[0]
            sampled_indices = np.unique(np.random.choice(inds, num_to_sample))
            all_sampled_indices.append(sampled_indices)
        indices = np.concatenate(all_sampled_indices)
        x_sampled = x[indices]
        y_sampled = y[indices]
        self.data = x_sampled
        self.targets = y_sampled

    def _subsample(self) -> None:
        """Subsamples the dataset to the specified fraction."""
        x = self.data
        y = self.targets
        num_samples = int(x.shape[0] * self.subsample_fraction)
        x_sampled = x[:num_samples]
        y_sampled = y[:num_samples]
        self.data = x_sampled
        self.targets = y_sampled

    def load_emnist_dataset(self) -> None:
        """Fetch the EMNIST dataset."""
        dataset = EMNIST(
            root=DATA_DIRNAME,
            split="byclass",
            train=self.train,
            download=False,
            transform=None,
            target_transform=None,
        )

        self.data = dataset.data
        self.targets = dataset.targets

        if self.sample_to_balance:
            self._sample_to_balance()

        if self.subsample_fraction is not None:
            self._subsample()


class EmnistDataLoaders:
    """Class for Emnist DataLoaders."""

    def __init__(
        self,
        splits: List[str],
        sample_to_balance: bool = False,
        subsample_fraction: float = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        batch_size: int = 128,
        shuffle: bool = False,
        num_workers: int = 0,
        cuda: bool = True,
        seed: int = 4711,
    ) -> None:
        """Fetches DataLoaders for given split(s).

        Args:
            splits (List[str]): One or both of the dataset splits "train" and "val".
            sample_to_balance (bool): If true, resamples the unbalanced if the split "byclass" is selected.
                Defaults to False.
            subsample_fraction (float): The fraction of the dataset will be loaded. If None or 0 the entire
                dataset will be loaded.
            transform (Optional[Callable]):  A function/transform that takes in an PIL image and returns a
                transformed version. E.g, transforms.RandomCrop. Defaults to None.
            target_transform (Optional[Callable]): A function/transform that takes in the target and
                transforms it. Defaults to None.
            batch_size (int): How many samples per batch to load. Defaults to 128.
            shuffle (bool): Set to True to have the data reshuffled at every epoch. Defaults to False.
            num_workers (int): How many subprocesses to use for data loading. 0 means that the data will be
                loaded in the main process. Defaults to 0.
            cuda (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning
                them. Defaults to True.
            seed (int): Seed for sampling.

        Raises:
            ValueError: If subsample_fraction is not None and outside the range (0, 1).

        """
        self.splits = splits

        if subsample_fraction is not None:
            if not 0.0 < subsample_fraction < 1.0:
                raise ValueError("The subsample fraction must be in (0, 1).")

        self.dataset_args = {
            "sample_to_balance": sample_to_balance,
            "subsample_fraction": subsample_fraction,
            "transform": transform,
            "target_transform": target_transform,
            "seed": seed,
        }
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.cuda = cuda
        self._data_loaders = self._load_data_loaders()

    def __repr__(self) -> str:
        """Returns information about the dataset."""
        return self._data_loaders[self.splits[0]].dataset.__repr__()

    @property
    def __name__(self) -> str:
        """Returns the name of the dataset."""
        return "Emnist"

    def __call__(self, split: str) -> DataLoader:
        """Returns the `split` DataLoader.

        Args:
            split (str): The dataset split, i.e. train or val.

        Returns:
            DataLoader: A PyTorch DataLoader.

        Raises:
            ValueError: If the split does not exist.

        """
        try:
            return self._data_loaders[split]
        except KeyError:
            raise ValueError(f"Split {split} does not exist.")

    def _load_data_loaders(self) -> Dict[str, DataLoader]:
        """Fetches the EMNIST dataset and return a Dict of PyTorch DataLoaders."""
        data_loaders = {}

        for split in ["train", "val"]:
            if split in self.splits:

                if split == "train":
                    self.dataset_args["train"] = True
                else:
                    self.dataset_args["train"] = False

                emnist_dataset = EmnistDataset(**self.dataset_args)

                emnist_dataset.load_emnist_dataset()

                data_loader = DataLoader(
                    dataset=emnist_dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.cuda,
                )

                data_loaders[split] = data_loader

        return data_loaders

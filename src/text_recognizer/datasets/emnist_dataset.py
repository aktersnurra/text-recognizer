"""Fetches a PyTorch DataLoader with the EMNIST dataset."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from loguru import logger
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor


DATA_DIRNAME = Path(__file__).resolve().parents[3] / "data"
ESSENTIALS_FILENAME = Path(__file__).resolve().parents[0] / "emnist_essentials.json"


class Transpose:
    """Transposes the EMNIST image to the correct orientation."""

    def __call__(self, image: Image) -> np.ndarray:
        """Swaps axis."""
        return np.array(image).swapaxes(0, 1)


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


def load_emnist_mapping() -> Dict[int, str]:
    """Load the EMNIST mapping."""
    with open(str(ESSENTIALS_FILENAME)) as f:
        essentials = json.load(f)
    return dict(essentials["mapping"])


class EmnistDataLoader:
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
        """Fetches DataLoaders.

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

        """
        self.splits = splits
        self.sample_to_balance = sample_to_balance
        if subsample_fraction is not None:
            assert (
                0.0 < subsample_fraction < 1.0
            ), " The subsample fraction must be in (0, 1)."
        self.subsample_fraction = subsample_fraction
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.cuda = cuda
        self._data_loaders = self._fetch_emnist_data_loaders()

    @property
    def __name__(self) -> str:
        """Returns the name of the dataset."""
        return "EMNIST"

    def __call__(self, split: str) -> Optional[DataLoader]:
        """Returns the `split` DataLoader.

        Args:
            split (str): The dataset split, i.e. train or val.

        Returns:
            type: A PyTorch DataLoader.

        Raises:
            ValueError: If the split does not exist.

        """
        try:
            return self._data_loaders[split]
        except KeyError:
            raise ValueError(f"Split {split} does not exist.")

    def _sample_to_balance(self, dataset: type = EMNIST) -> EMNIST:
        """Because the dataset is not balanced, we take at most the mean number of instances per class."""
        np.random.seed(self.seed)
        x = dataset.data
        y = dataset.targets
        num_to_sample = int(np.bincount(y.flatten()).mean())
        all_sampled_indices = []
        for label in np.unique(y.flatten()):
            inds = np.where(y == label)[0]
            sampled_indices = np.unique(np.random.choice(inds, num_to_sample))
            all_sampled_indices.append(sampled_indices)
        indices = np.concatenate(all_sampled_indices)
        x_sampled = x[indices]
        y_sampled = y[indices]
        dataset.data = x_sampled
        dataset.targets = y_sampled

        return dataset

    def _subsample(self, dataset: type = EMNIST) -> EMNIST:
        """Subsamples the dataset to the specified fraction."""
        x = dataset.data
        y = dataset.targets
        num_samples = int(x.shape[0] * self.subsample_fraction)
        x_sampled = x[:num_samples]
        y_sampled = y[:num_samples]
        dataset.data = x_sampled
        dataset.targets = y_sampled

        return dataset

    def _fetch_emnist_dataset(self, train: bool) -> EMNIST:
        """Fetch the EMNIST dataset."""
        if self.transform is None:
            transform = Compose([Transpose(), ToTensor()])

        dataset = EMNIST(
            root=DATA_DIRNAME,
            split="byclass",
            train=train,
            download=False,
            transform=transform,
            target_transform=self.target_transform,
        )

        if self.sample_to_balance:
            dataset = self._sample_to_balance(dataset)

        if self.subsample_fraction is not None:
            dataset = self._subsample(dataset)

        return dataset

    def _fetch_emnist_data_loaders(self) -> Dict[str, DataLoader]:
        """Fetches the EMNIST dataset and return a Dict of PyTorch DataLoaders."""
        data_loaders = {}

        for split in ["train", "val"]:
            if split in self.splits:

                if split == "train":
                    train = True
                else:
                    train = False

                dataset = self._fetch_emnist_dataset(train)

                data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.cuda,
                )

                data_loaders[split] = data_loader

        return data_loaders

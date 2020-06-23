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


def save_emnist_essentials(emnsit_dataset: EMNIST) -> None:
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


def load_emnist_mapping() -> Dict:
    """Load the EMNIST mapping."""
    with open(str(ESSENTIALS_FILENAME)) as f:
        essentials = json.load(f)
    return dict(essentials["mapping"])


def _sample_to_balance(dataset: EMNIST, seed: int = 4711) -> None:
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(seed)
    x = dataset.data
    y = dataset.targets
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    dataset.data = x_sampled
    dataset.targets = y_sampled


def fetch_emnist_dataset(
    split: str,
    train: bool,
    sample_to_balance: bool = False,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
) -> EMNIST:
    """Fetch the EMNIST dataset."""
    if transform is None:
        transform = Compose([Transpose(), ToTensor()])

    dataset = EMNIST(
        root=DATA_DIRNAME,
        split="byclass",
        train=train,
        download=False,
        transform=transform,
        target_transform=target_transform,
    )

    if sample_to_balance and split == "byclass":
        _sample_to_balance(dataset)

    return dataset


def fetch_emnist_data_loader(
    splits: List[str],
    sample_to_balance: bool = False,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    cuda: bool = True,
) -> Dict[DataLoader]:
    """Fetches the EMNIST dataset and return a PyTorch DataLoader.

    Args:
        splits (List[str]): One or both of the dataset splits "train" and "val".
        sample_to_balance (bool): If true, resamples the unbalanced if the split "byclass" is selected.
            Defaults to False.
        transform (Optional[Callable]):  A function/transform that takes in an PIL image and returns a
            transformed version. E.g, transforms.RandomCrop. Defaults to None.
        target_transform (Optional[Callable]): A function/transform that takes in the target and transforms
            it.
            Defaults to None.
        batch_size (int): How many samples per batch to load. Defaults to 128.
        shuffle (bool): Set to True to have the data reshuffled at every epoch. Defaults to False.
        num_workers (int): How many subprocesses to use for data loading. 0 means that the data will be
            loaded in the main process. Defaults to 0.
        cuda (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning
            them. Defaults to True.

    Returns:
        Dict: A dict containing PyTorch DataLoader(s) with emnist characters.

    """
    data_loaders = {}

    for split in ["train", "val"]:
        if split in splits:

            if split == "train":
                train = True
            else:
                train = False

            dataset = fetch_emnist_dataset(
                split, train, sample_to_balance, transform, target_transform
            )

            data_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=cuda,
            )

            data_loaders[split] = data_loader

    return data_loaders

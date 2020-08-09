"""Util functions for datasets."""
import importlib
from typing import Callable, Dict, List, Type

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class Transpose:
    """Transposes the EMNIST image to the correct orientation."""

    def __call__(self, image: Image) -> np.ndarray:
        """Swaps axis."""
        return np.array(image).swapaxes(0, 1)


def fetch_data_loaders(
    splits: List[str],
    dataset: str,
    dataset_args: Dict,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    cuda: bool = True,
) -> Dict[str, DataLoader]:
    """Fetches DataLoaders for given split(s) as a dictionary.

    Loads the dataset class given, and loads it with the dataset arguments, for the number of splits specified. Then
    calls the DataLoader. Added to a dictionary with the split as key and DataLoader as value.

    Args:
        splits (List[str]): One or both of the dataset splits "train" and "val".
        dataset (str): The name of the dataset.
        dataset_args (Dict): The dataset arguments.
        batch_size (int): How many samples per batch to load. Defaults to 128.
        shuffle (bool): Set to True to have the data reshuffled at every epoch. Defaults to False.
        num_workers (int): How many subprocesses to use for data loading. 0 means that the data will be
            loaded in the main process. Defaults to 0.
        cuda (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning
            them. Defaults to True.

    Returns:
        Dict[str, DataLoader]: Dictionary with split as key and PyTorch DataLoader as value.

    """

    def check_dataset_args(args: Dict, split: str) -> Dict:
        args["train"] = True if split == "train" else False
        return args

    # Import dataset module.
    datasets_module = importlib.import_module("text_recognizer.datasets")
    dataset_ = getattr(datasets_module, dataset)

    data_loaders = {}

    for split in ["train", "val"]:
        if split in splits:

            data_loader = DataLoader(
                dataset=dataset_(**check_dataset_args(dataset_args, split)),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=cuda,
            )

            data_loaders[split] = data_loader

    return data_loaders

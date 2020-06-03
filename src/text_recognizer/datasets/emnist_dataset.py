"""Fetches a PyTorch DataLoader with the EMNIST dataset."""
from pathlib import Path
from typing import Callable

import click
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST


@click.command()
@click.option("--split", "-s", default="byclass")
def download_emnist(split: str) -> None:
    """Download the EMNIST dataset via the PyTorch class."""
    data_dir = Path(__file__).resolve().parents[3] / "data"
    logger.debug(f"Data directory is: {data_dir}")
    EMNIST(root=data_dir, split=split, download=True)


def fetch_dataloader(
    root: str,
    split: str,
    train: bool,
    download: bool,
    transform: Callable = None,
    target_transform: Callable = None,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    cuda: bool = True,
) -> DataLoader:
    """Down/load the EMNIST dataset and return a PyTorch DataLoader.

    Args:
        root (str): Root directory of dataset where EMNIST/processed/training.pt and EMNIST/processed/test.pt
            exist.
        split (str): The dataset has 6 different splits: byclass, bymerge, balanced, letters, digits and mnist.
            This argument specifies which one to use.
        train (bool): If True, creates dataset from training.pt, otherwise from test.pt.
        download (bool): If true, downloads the dataset from the internet and puts it in root directory. If
            dataset is already downloaded, it is not downloaded again.
        transform (Callable): A function/transform that takes in an PIL image and returns a transformed version.
            E.g, transforms.RandomCrop.
        target_transform (Callable): A function/transform that takes in the target and transforms it.
        batch_size (int): How many samples per batch to load (the default is 128).
        shuffle (bool): Set to True to have the data reshuffled at every epoch (the default is False).
        num_workers (int): How many subprocesses to use for data loading. 0 means that the data will be loaded in
            the main process (default: 0).
        cuda (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

    Returns:
        DataLoader: A PyTorch DataLoader with emnist characters.

    """
    dataset = EMNIST(
        root=root,
        split=split,
        train=train,
        download=download,
        transform=transform,
        target_transform=target_transform,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cuda,
    )

    return data_loader

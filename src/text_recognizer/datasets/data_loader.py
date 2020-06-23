"""Data loader collection."""

from typing import Dict

from torch.utils.data import DataLoader

from text_recognizer.datasets.emnist_dataset import fetch_emnist_data_loader


def fetch_data_loader(data_loader_args: Dict) -> DataLoader:
    """Fetches the specified PyTorch data loader."""
    if data_loader_args.pop("name").lower() == "emnist":
        return fetch_emnist_data_loader(data_loader_args)
    else:
        raise NotImplementedError

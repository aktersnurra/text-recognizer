"""EMNIST dataset: downloads it from FSDL aws url if not present."""
from pathlib import Path
from typing import Sequence, Tuple
import json
import os
import shutil
import zipfile

import h5py
import numpy as np
from loguru import logger
import toml
import torch
from torch.utils.data import random_split
from torchvision import transforms

from text_recognizer.datasets.base_dataset import BaseDataset
from text_recognizer.datasets.base_data_module import BaseDataModule, load_print_info
from text_recognizer.datasets.download_utils import download_dataset


SEED = 4711
NUM_SPECIAL_TOKENS = 4
SAMPLE_TO_BALANCE = True 

RAW_DATA_DIRNAME = BaseDataModule.data_dirname() / "raw" / "emnist"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded" / "emnist"
PROCESSED_DATA_DIRNAME = BaseDataset.data_dirname() / "processed" / "emnist"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
ESSENTIALS_FILENAME = Path(__file__).parents[0].resolve() / "emnsit_essentials.json"


class EMNIST(BaseDataModule):
    """
    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self, batch_size: int = 128, num_workers: int = 0, train_fraction: float = 0.8) -> None:
        super().__init__(batch_size, num_workers)
        if not ESSENTIALS_FILENAME.exists():
            _download_and_process_emnist()
        with ESSENTIALS_FILENAME.open() as f:
            essentials = json.load(f)
        self.train_fraction = train_fraction
        self.mapping = list(essentials["characters"])
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, *essentials["input_shape"])
        self.output_dims = (1,)

    def prepare_data(self) -> None:
        if not PROCESSED_DATA_FILENAME.exists():
            _download_and_process_emnist()

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                data = f["x_train"][:]
                targets = f["y_train"][:]
        
            dataset_train = BaseDataset(data, targets, transform=self.transform)
            train_size = int(self.train_fraction * len(dataset_train))
            val_size = len(dataset_train) - train_size
            self.data_train, self.data_val = random_split(dataset_train, [train_size, val_size], generator=torch.Generator())

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                data = f["x_test"][:]
                targets = f["y_test"][:]
            self.data_test = BaseDataset(data, targets, transform=self.transform)


    def __repr__(self) -> str:
        basic = f"EMNIST Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.dims}\n"
        if not any([self.data_train, self.data_val, self.data_test]):
            return basic

        datum, target = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(datum.shape, datum.dtype, datum.min(), datum.mean(), datum.std(), datum.max())}\n"
            f"Batch y stats: {(target.shape, target.dtype, target.min(), target.max())}\n"
        )

        return basic + data


def _download_and_process_emnist() -> None:
    metadata = toml.load(METADATA_FILENAME)
    download_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path) -> None:
    logger.info("Unzipping EMNIST...")
    curdir = os.getcwd()
    os.chdir(dirname)
    content = zipfile.ZipFile(filename, "r")
    content.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat

    logger.info("Loading training data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
    x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS

    if SAMPLE_TO_BALANCE:
        logger.info("Balancing classes to reduce amount of data")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)


    logger.info("Saving to HDF5 in a compressed format...")
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    logger.info("Saving essential dataset parameters to text_recognizer/datasets...")
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
    characters = _augment_emnist_characters(mapping.values())
    essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}

    with ESSENTIALS_FILENAME.open(mode="w") as f:
        json.dump(essentials, f)

    logger.info("Cleaning up...")
    shutil.rmtree("matlab")
    os.chdir(curdir)


def _sample_to_balance(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Balances the dataset by taking the mean number of instances per class."""
    np.random.seed(SEED)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_indices = []
    for label in np.unique(y.flatten()):
        indices = np.where(y == label)[0]
        sampled_indices = np.unique(np.random.choice(indices, num_to_sample))
        all_sampled_indices.append(sampled_indices)
    indices = np.concatenate(all_sampled_indices)
    x_sampled = x[indices]
    y_sampled= y[indices]
    return x_sampled, y_sampled


def _augment_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    """Augment the mapping with extra symbols."""
    # Extra characters from the IAM dataset.
    iam_characters = [
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

    # Also add special tokens for:
    # - CTC blank token at index 0
    # - Start token at index 1
    # - End token at index 2
    # - Padding token at index 3
    # Note: Do not forget to update NUM_SPECIAL_TOKENS if changing this!
    return ["<b>", "<s>", "</s>", "<p>", *characters, *iam_characters]


if __name__ == "__main__":
    load_print_info(EMNIST)

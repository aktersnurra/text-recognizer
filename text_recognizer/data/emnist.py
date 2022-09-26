"""EMNIST dataset: downloads it from FSDL aws url if not present."""
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import h5py
import numpy as np
import toml
from loguru import logger as log

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.base_dataset import BaseDataset, split_dataset
from text_recognizer.data.transforms.load_transform import load_transform_from_file
from text_recognizer.data.utils.download_utils import download_dataset
from text_recognizer.metadata import emnist as metadata


class EMNIST(BaseDataModule):
    """Lightning DataModule class for loading EMNIST dataset.

    'The EMNIST dataset is a set of handwritten character digits derived from the NIST
    Special Database 19 and converted to a 28x28 pixel image format and dataset
    structure that directly matches the MNIST dataset.'
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dims = (1, *self.mapping.input_size)

    def prepare_data(self) -> None:
        """Downloads dataset if not present."""
        if not metadata.PROCESSED_DATA_FILENAME.exists():
            download_and_process_emnist()

    def setup(self, stage: Optional[str] = None) -> None:
        """Loads the dataset specified by the stage."""
        if stage == "fit" or stage is None:
            with h5py.File(metadata.PROCESSED_DATA_FILENAME, "r") as f:
                self.x_train = f["x_train"][:]
                self.y_train = f["y_train"][:].squeeze().astype(int)

            dataset_train = BaseDataset(
                self.x_train, self.y_train, transform=self.transform
            )
            self.data_train, self.data_val = split_dataset(
                dataset_train, fraction=self.train_fraction, seed=metadata.SEED
            )

        if stage == "test" or stage is None:
            with h5py.File(metadata.PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = BaseDataset(
                self.x_test, self.y_test, transform=self.transform
            )

    def __repr__(self) -> str:
        """Returns string with info about the dataset."""
        basic = (
            "EMNIST Dataset\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Mapping: {self.mapping}\n"
            f"Dims: {self.dims}\n"
        )
        if not any([self.data_train, self.data_val, self.data_test]):
            return basic

        datum, target = next(iter(self.train_dataloader()))
        data = (
            "Train/val/test sizes: "
            f"{len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            "Batch x stats: "
            f"{(datum.shape, datum.dtype, datum.min())}"
            f"{(datum.mean(), datum.std(), datum.max())}\n"
            f"Batch y stats: "
            f"{(target.shape, target.dtype, target.min(), target.max())}\n"
        )

        return basic + data


def download_and_process_emnist() -> None:
    """Downloads and preprocesses EMNIST dataset."""
    metadata_ = toml.load(metadata.METADATA_FILENAME)
    download_dataset(metadata_, metadata.DL_DATA_DIRNAME)
    _process_raw_dataset(metadata_["filename"], metadata.DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path) -> None:
    """Processes the raw EMNIST dataset."""
    log.info("Unzipping EMNIST...")
    curdir = os.getcwd()
    os.chdir(dirname)
    content = zipfile.ZipFile(filename, "r")
    content.extract("matlab/emnist-byclass.mat")

    from scipy.io import loadmat

    log.info("Loading training data from .mat file")
    data = loadmat("matlab/emnist-byclass.mat")
    x_train = (
        data["dataset"]["train"][0, 0]["images"][0, 0]
        .reshape(-1, 28, 28)
        .swapaxes(1, 2)
    )
    y_train = (
        data["dataset"]["train"][0, 0]["labels"][0, 0] + metadata.NUM_SPECIAL_TOKENS
    )
    x_test = (
        data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
    )
    y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + metadata.NUM_SPECIAL_TOKENS

    if metadata.SAMPLE_TO_BALANCE:
        log.info("Balancing classes to reduce amount of data")
        x_train, y_train = _sample_to_balance(x_train, y_train)
        x_test, y_test = _sample_to_balance(x_test, y_test)

    log.info("Saving to HDF5 in a compressed format...")
    metadata.PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    with h5py.File(metadata.PROCESSED_DATA_FILENAME, "w") as f:
        f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
        f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
        f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
        f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

    log.info("Saving essential dataset parameters to text_recognizer/datasets...")
    mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
    characters = _augment_emnist_characters(mapping.values())
    essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}

    with metadata.ESSENTIALS_FILENAME.open(mode="w") as f:
        json.dump(essentials, f)

    log.info("Cleaning up...")
    shutil.rmtree("matlab")
    os.chdir(curdir)


def _sample_to_balance(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Balances the dataset by taking the mean number of instances per class."""
    np.random.seed(metadata.SEED)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_indices = []
    for label in np.unique(y.flatten()):
        indices = np.where(y == label)[0]
        sampled_indices = np.unique(np.random.choice(indices, num_to_sample))
        all_sampled_indices.append(sampled_indices)
    indices = np.concatenate(all_sampled_indices)
    x_sampled = x[indices]
    y_sampled = y[indices]
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
    return ["<b>", "<s>", "<e>", "<p>", *characters, *iam_characters]


def download_emnist() -> None:
    """Download dataset from internet, if it does not exists, and displays info."""
    transform = load_transform_from_file("transform/default.yaml")
    load_and_print_info(EMNIST(transform=transform, test_transfrom=transform))

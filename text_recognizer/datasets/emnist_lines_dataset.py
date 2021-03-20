"""Emnist Lines dataset: synthetic handwritten lines dataset made from Emnist characters."""

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import click
import h5py
from loguru import logger
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from text_recognizer.datasets.dataset import Dataset
from text_recognizer.datasets.emnist_dataset import EmnistDataset, Transpose
from text_recognizer.datasets.sentence_generator import SentenceGenerator
from text_recognizer.datasets.util import (
    DATA_DIRNAME,
    EmnistMapper,
    ESSENTIALS_FILENAME,
)

DATA_DIRNAME = DATA_DIRNAME / "processed" / "emnist_lines"

MAX_WIDTH = 952


class EmnistLinesDataset(Dataset):
    """Synthetic dataset of lines from the Brown corpus with Emnist characters."""

    def __init__(
        self,
        train: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        subsample_fraction: float = None,
        max_length: int = 34,
        min_overlap: float = 0,
        max_overlap: float = 0.33,
        num_samples: int = 10000,
        seed: int = 4711,
        init_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        lower: bool = False,
    ) -> None:
        """Set attributes and loads the dataset.

        Args:
            train (bool): Flag for the filename. Defaults to False. Defaults to None.
            transform (Optional[Callable]): The transform of the data. Defaults to None.
            target_transform (Optional[Callable]): The transform of the target. Defaults to None.
            subsample_fraction (float): The fraction of the dataset to use for training. Defaults to None.
            max_length (int): The maximum number of characters. Defaults to 34.
            min_overlap (float): The minimum overlap between concatenated images. Defaults to 0.
            max_overlap (float): The maximum overlap between concatenated images. Defaults to 0.33.
            num_samples (int): Number of samples to generate. Defaults to 10000.
            seed (int): Seed number. Defaults to 4711.
            init_token (Optional[str]): String representing the start of sequence token. Defaults to None.
            pad_token (Optional[str]): String representing the pad token. Defaults to None.
            eos_token (Optional[str]): String representing the end of sequence token. Defaults to None.
            lower (bool): If True, convert uppercase letters to lowercase. Otherwise, use both upper and lowercase.

        """
        self.pad_token = "_" if pad_token is None else pad_token

        super().__init__(
            train=train,
            transform=transform,
            target_transform=target_transform,
            subsample_fraction=subsample_fraction,
            init_token=init_token,
            pad_token=self.pad_token,
            eos_token=eos_token,
            lower=lower,
        )

        # Extract dataset information.
        self._input_shape = self._mapper.input_shape
        self.num_classes = self._mapper.num_classes

        self.max_length = max_length
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.num_samples = num_samples
        self._input_shape = (
            self.input_shape[0],
            self.input_shape[1] * self.max_length,
        )
        self._output_shape = (self.max_length, self.num_classes)
        self.seed = seed

        # Placeholders for the dataset.
        self._data = None
        self._target = None

    def __getitem__(self, index: Union[int, Tensor]) -> Tuple[Tensor, Tensor]:
        """Fetches data, target pair of the dataset for a given and index or indices.

        Args:
            index (Union[int, Tensor]): Either a list or int of indices/index.

        Returns:
            Tuple[Tensor, Tensor]: Data target pair.

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
            "EMNIST Lines Dataset\n"  # pylint: disable=no-member
            f"Max length: {self.max_length}\n"
            f"Min overlap: {self.min_overlap}\n"
            f"Max overlap: {self.max_overlap}\n"
            f"Num classes: {self.num_classes}\n"
            f"Input shape: {self.input_shape}\n"
            f"Data: {self.data.shape}\n"
            f"Tagets: {self.targets.shape}\n"
        )

    @property
    def data_filename(self) -> Path:
        """Path to the h5 file."""
        filename = "train.pt" if self.train else "test.pt"
        return DATA_DIRNAME / filename

    def load_or_generate_data(self) -> None:
        """Loads the dataset, if it does not exist a new dataset is generated before loading it."""
        np.random.seed(self.seed)

        if not self.data_filename.exists():
            self._generate_data()
        self._load_data()
        self._subsample()

    def _load_data(self) -> None:
        """Loads the dataset from the h5 file."""
        logger.debug("EmnistLinesDataset loading data from HDF5...")
        with h5py.File(self.data_filename, "r") as f:
            self._data = f["data"][()]
            self._targets = f["targets"][()]

    def _generate_data(self) -> str:
        """Generates a dataset with the Brown corpus and Emnist characters."""
        logger.debug("Generating data...")

        sentence_generator = SentenceGenerator(self.max_length)

        # Load emnist dataset.
        emnist = EmnistDataset(
            train=self.train, sample_to_balance=True, pad_token=self.pad_token
        )
        emnist.load_or_generate_data()

        samples_by_character = get_samples_by_character(
            emnist.data.numpy(), emnist.targets.numpy(), self.mapper.mapping,
        )

        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_filename, "a") as f:
            data, targets = create_dataset_of_images(
                self.num_samples,
                samples_by_character,
                sentence_generator,
                self.min_overlap,
                self.max_overlap,
            )

            targets = convert_strings_to_categorical_labels(
                targets, emnist.inverse_mapping
            )

            f.create_dataset("data", data=data, dtype="u1", compression="lzf")
            f.create_dataset("targets", data=targets, dtype="u1", compression="lzf")


def get_samples_by_character(
    samples: np.ndarray, labels: np.ndarray, mapping: Dict
) -> defaultdict:
    """Creates a dictionary with character as key and value as the list of images of that character.

    Args:
        samples (np.ndarray): Dataset of images of characters.
        labels (np.ndarray): The labels for each image.
        mapping (Dict): The Emnist mapping dictionary.

    Returns:
        defaultdict: A dictionary with characters as keys and list of images as values.

    """
    samples_by_character = defaultdict(list)
    for sample, label in zip(samples, labels.flatten()):
        samples_by_character[mapping[label]].append(sample)
    return samples_by_character


def select_letter_samples_for_string(
    string: str, samples_by_character: Dict
) -> List[np.ndarray]:
    """Randomly selects Emnist characters to use for the senetence.

    Args:
        string (str): The word or sentence.
        samples_by_character (Dict): The dictionary of emnist images of each character.

    Returns:
        List[np.ndarray]: A list of emnist images of the string.

    """
    zero_image = np.zeros((28, 28), np.uint8)
    sample_image_by_character = {}
    for character in string:
        if character in sample_image_by_character:
            continue
        samples = samples_by_character[character]
        sample = samples[np.random.choice(len(samples))] if samples else zero_image
        sample_image_by_character[character] = sample.reshape(28, 28).swapaxes(0, 1)
    return [sample_image_by_character[character] for character in string]


def construct_image_from_string(
    string: str, samples_by_character: Dict, min_overlap: float, max_overlap: float
) -> np.ndarray:
    """Concatenates images of the characters in the string.

    The concatination is made with randomly selected overlap so that some portion of the character will overlap.

    Args:
        string (str): The word or sentence.
        samples_by_character (Dict): The dictionary of emnist images of each character.
        min_overlap (float): Minimum amount of overlap between Emnist images.
        max_overlap (float): Maximum amount of overlap between Emnist images.

    Returns:
        np.ndarray: The Emnist image of the string.

    """
    overlap = np.random.uniform(min_overlap, max_overlap)
    sampled_images = select_letter_samples_for_string(string, samples_by_character)
    length = len(sampled_images)
    height, width = sampled_images[0].shape
    next_overlap_width = width - int(overlap * width)
    concatenated_image = np.zeros((height, width * length), np.uint8)
    x = 0
    for image in sampled_images:
        concatenated_image[:, x : (x + width)] += image
        x += next_overlap_width

    if concatenated_image.shape[-1] > MAX_WIDTH:
        concatenated_image = Tensor(concatenated_image).unsqueeze(0)
        concatenated_image = F.interpolate(
            concatenated_image, size=MAX_WIDTH, mode="nearest"
        )
        concatenated_image = concatenated_image.squeeze(0).numpy()

    return np.minimum(255, concatenated_image)


def create_dataset_of_images(
    length: int,
    samples_by_character: Dict,
    sentence_generator: SentenceGenerator,
    min_overlap: float,
    max_overlap: float,
) -> Tuple[np.ndarray, List[str]]:
    """Creates a dataset with images and labels from strings generated from the SentenceGenerator.

    Args:
        length (int): The number of characters for each string.
        samples_by_character (Dict): The dictionary of emnist images of each character.
        sentence_generator (SentenceGenerator): A SentenceGenerator objest.
        min_overlap (float): Minimum amount of overlap between Emnist images.
        max_overlap (float): Maximum amount of overlap between Emnist images.

    Returns:
        Tuple[np.ndarray, List[str]]: A list of Emnist images and a list of the strings (labels).

    Raises:
        RuntimeError: If the sentence generator is not able to generate a string.

    """
    sample_label = sentence_generator.generate()
    sample_image = construct_image_from_string(sample_label, samples_by_character, 0, 0)
    images = np.zeros((length, sample_image.shape[0], sample_image.shape[1]), np.uint8)
    labels = []
    for n in range(length):
        label = None
        # Try several times to generate before actually throwing an error.
        for _ in range(10):
            try:
                label = sentence_generator.generate()
                break
            except Exception:  # pylint: disable=broad-except
                pass
        if label is None:
            raise RuntimeError("Was not able to generate a valid string.")
        images[n] = construct_image_from_string(
            label, samples_by_character, min_overlap, max_overlap
        )
        labels.append(label)
    return images, labels


def convert_strings_to_categorical_labels(
    labels: List[str], mapping: Dict
) -> np.ndarray:
    """Translates a string of characters in to a target array of class int."""
    return np.array([[mapping[c] for c in label] for label in labels])


@click.command()
@click.option(
    "--max_length", type=int, default=34, help="Number of characters in a sentence."
)
@click.option(
    "--min_overlap", type=float, default=0.0, help="Min overlap between characters."
)
@click.option(
    "--max_overlap", type=float, default=0.33, help="Max overlap between characters."
)
@click.option("--num_train", type=int, default=10_000, help="Number of train examples.")
@click.option("--num_test", type=int, default=1_000, help="Number of test examples.")
def create_datasets(
    max_length: int = 34,
    min_overlap: float = 0,
    max_overlap: float = 0.33,
    num_train: int = 10000,
    num_test: int = 1000,
) -> None:
    """Creates a training an validation dataset of Emnist lines."""
    num_samples = [num_train, num_test]
    for num, train in zip(num_samples, [True, False]):
        emnist_lines = EmnistLinesDataset(
            train=train,
            max_length=max_length,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            num_samples=num,
        )
        emnist_lines.load_or_generate_data()


if __name__ == "__main__":
    create_datasets()

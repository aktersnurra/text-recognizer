"""Dataset of generated text from EMNIST characters."""
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Tuple

import attr
import h5py
from loguru import logger as log
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from text_recognizer.data.base_data_module import (
    BaseDataModule,
    load_and_print_info,
)
from text_recognizer.data.base_dataset import BaseDataset, convert_strings_to_labels
from text_recognizer.data.emnist import EMNIST
from text_recognizer.data.sentence_generator import SentenceGenerator


DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist_lines"
ESSENTIALS_FILENAME = (
    Path(__file__).parents[0].resolve() / "emnist_lines_essentials.json"
)

SEED = 4711
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 1024
IMAGE_X_PADDING = 28
MAX_OUTPUT_LENGTH = 89  # Same as IAMLines


@attr.s(auto_attribs=True, repr=False)
class EMNISTLines(BaseDataModule):
    """EMNIST Lines dataset: synthetic handwritten lines dataset made from EMNIST."""

    augment: bool = attr.ib(default=True)
    max_length: int = attr.ib(default=128)
    min_overlap: float = attr.ib(default=0.0)
    max_overlap: float = attr.ib(default=0.33)
    num_train: int = attr.ib(default=10_000)
    num_val: int = attr.ib(default=2_000)
    num_test: int = attr.ib(default=2_000)
    emnist: EMNIST = attr.ib(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        """Post init constructor."""
        self.emnist = EMNIST(mapping=self.mapping)

        max_width = (
            int(self.emnist.dims[2] * (self.max_length + 1) * (1 - self.min_overlap))
            + IMAGE_X_PADDING
        )

        if max_width >= IMAGE_WIDTH:
            raise ValueError(
                f"max_width {max_width} greater than IMAGE_WIDTH {IMAGE_WIDTH}"
            )

        self.dims = (self.emnist.dims[0], IMAGE_HEIGHT, IMAGE_WIDTH)

        if self.max_length >= MAX_OUTPUT_LENGTH:
            raise ValueError("max_length greater than MAX_OUTPUT_LENGTH")

        self.output_dims = (MAX_OUTPUT_LENGTH, 1)

    @property
    def data_filename(self) -> Path:
        """Return name of dataset."""
        return DATA_DIRNAME / (
            f"ml_{self.max_length}_"
            f"o{self.min_overlap:f}_{self.max_overlap:f}_"
            f"ntr{self.num_train}_"
            f"ntv{self.num_val}_"
            f"nte{self.num_test}.h5"
        )

    def prepare_data(self) -> None:
        """Prepare the dataset."""
        if self.data_filename.exists():
            return
        np.random.seed(SEED)
        self._generate_data("train")
        self._generate_data("val")
        self._generate_data("test")

    def setup(self, stage: str = None) -> None:
        """Loads the dataset."""
        log.info("EMNISTLinesDataset loading data from HDF5...")
        if stage == "fit" or stage is None:
            print(self.data_filename)
            with h5py.File(self.data_filename, "r") as f:
                x_train = f["x_train"][:]
                y_train = torch.LongTensor(f["y_train"][:])
                x_val = f["x_val"][:]
                y_val = torch.LongTensor(f["y_val"][:])

            self.data_train = BaseDataset(
                x_train, y_train, transform=_get_transform(augment=self.augment)
            )
            self.data_val = BaseDataset(
                x_val, y_val, transform=_get_transform(augment=self.augment)
            )

        if stage == "test" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_test = f["x_test"][:]
                y_test = torch.LongTensor(f["y_test"][:])

            self.data_test = BaseDataset(
                x_test, y_test, transform=_get_transform(augment=False)
            )

    def __repr__(self) -> str:
        """Return str about dataset."""
        basic = (
            "EMNISTLines2 Dataset\n"  # pylint: disable=no-member
            f"Min overlap: {self.min_overlap}\n"
            f"Max overlap: {self.max_overlap}\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )

        if not any([self.data_train, self.data_val, self.data_test]):
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            "Train/val/test sizes: "
            f"{len(self.data_train)}, "
            f"{len(self.data_val)}, "
            f"{len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

    def _generate_data(self, split: str) -> None:
        log.info(f"EMNISTLines generating data for {split}...")
        sentence_generator = SentenceGenerator(
            self.max_length - 2
        )  # Subtract by 2 because start/end token

        emnist = self.emnist
        emnist.prepare_data()
        emnist.setup()

        if split == "train":
            samples_by_char = _get_samples_by_char(
                emnist.x_train, emnist.y_train, self.mapping.mapping
            )
            num = self.num_train
        elif split == "val":
            samples_by_char = _get_samples_by_char(
                emnist.x_train, emnist.y_train, self.mapping.mapping
            )
            num = self.num_val
        else:
            samples_by_char = _get_samples_by_char(
                emnist.x_test, emnist.y_test, self.mapping.mapping
            )
            num = self.num_test

        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_filename, "a") as f:
            x, y = _create_dataset_of_images(
                num,
                samples_by_char,
                sentence_generator,
                self.min_overlap,
                self.max_overlap,
                self.dims,
            )
            y = convert_strings_to_labels(
                y, self.mapping.inverse_mapping, length=MAX_OUTPUT_LENGTH
            )
            f.create_dataset(f"x_{split}", data=x, dtype="u1", compression="lzf")
            f.create_dataset(f"y_{split}", data=y, dtype="u1", compression="lzf")


def _get_samples_by_char(
    samples: np.ndarray, labels: np.ndarray, mapping: List
) -> defaultdict:
    samples_by_char = defaultdict(list)
    for sample, label in zip(samples, labels):
        samples_by_char[mapping[label]].append(sample)
    return samples_by_char


def _select_letter_samples_for_string(
    string: str, samples_by_char: defaultdict
) -> List[Tensor]:
    null_image = torch.zeros((28, 28), dtype=torch.uint8)
    sample_image_by_char = {}
    for char in string:
        if char in sample_image_by_char:
            continue
        samples = samples_by_char[char]
        sample = samples[np.random.choice(len(samples))] if samples else null_image
        sample_image_by_char[char] = sample.reshape(28, 28)
    return [sample_image_by_char[char] for char in string]


def _construct_image_from_string(
    string: str,
    samples_by_char: defaultdict,
    min_overlap: float,
    max_overlap: float,
    width: int,
) -> Tensor:
    overlap = np.random.uniform(min_overlap, max_overlap)
    sampled_images = _select_letter_samples_for_string(string, samples_by_char)
    H, W = sampled_images[0].shape
    next_overlap_width = W - int(overlap * W)
    concatenated_image = torch.zeros((H, width), dtype=torch.uint8)
    x = IMAGE_X_PADDING
    for image in sampled_images:
        concatenated_image[:, x : (x + W)] += image
        x += next_overlap_width
    return torch.minimum(Tensor([255]), concatenated_image)


def _create_dataset_of_images(
    num_samples: int,
    samples_by_char: defaultdict,
    sentence_generator: SentenceGenerator,
    min_overlap: float,
    max_overlap: float,
    dims: Tuple,
) -> Tuple[Tensor, Tensor]:
    images = torch.zeros((num_samples, IMAGE_HEIGHT, dims[2]))
    labels = []
    for n in range(num_samples):
        label = sentence_generator.generate()
        crop = _construct_image_from_string(
            label, samples_by_char, min_overlap, max_overlap, dims[-1]
        )
        height = crop.shape[0]
        y = (IMAGE_HEIGHT - height) // 2
        images[n, y : (y + height), :] = crop
        labels.append(label)
    return images, labels


def _get_transform(augment: bool = False) -> Callable:
    if not augment:
        return transforms.Compose([transforms.ToTensor()])
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=(0.5, 1.0)),
            transforms.RandomAffine(
                degrees=3,
                translate=(0.0, 0.05),
                scale=(0.4, 1.1),
                shear=(-40, 50),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
        ]
    )


def generate_emnist_lines() -> None:
    """Generates a synthetic handwritten dataset and displays info."""
    load_and_print_info(EMNISTLines)

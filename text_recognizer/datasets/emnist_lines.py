"""Dataset of generated text from EMNIST characters."""
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence

import h5py
from loguru import logger
import numpy as np
import torch
from torchvision import transforms

from text_recognizer.datasets.base_dataset import BaseDataset
from text_recognizer.datasets.base_data_module import BaseDataModule
from text_recognizer.datasets.emnist import EMNIST
from text_recognizer.datasets.sentence_generator import SentenceGenerator


DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "emnist_lines"
ESSENTIALS_FILENAME = (
    Path(__file__).parents[0].resolve() / "emnist_lines_essentials.json"
)

SEED = 4711
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 1024
IMAGE_X_PADDING = 28
MAX_OUTPUT_LENGTH = 89  # Same as IAMLines


class EMNISTLines(BaseDataModule):
    """EMNIST Lines dataset: synthetic handwritten lines dataset made from EMNIST,"""

    def __init__(
        self,
        augment: bool = True,
        batch_size: int = 128,
        num_workers: int = 0,
        max_length: int = 32,
        min_overlap: float = 0.0,
        max_overlap: float = 0.33,
        num_train: int = 10_000,
        num_val: int = 2_000,
        num_test: int = 2_000,
    ) -> None:
        super().__init__(batch_size, num_workers)

        self.augment = augment
        self.max_length = max_length
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

        self.emnist = EMNIST()
        self.mapping = self.emnist.mapping
        max_width = int(self.emnist.dims[2] * (self.max_length + 1) * (1 - self.min_overlap)) + IMAGE_X_PADDING
        
        if max_width <= IMAGE_WIDTH:
            raise ValueError("max_width greater than IMAGE_WIDTH")

        self.dims = (
            self.emnist.dims[0],
            self.emnist.dims[1],
            self.emnist.dims[2] * self.max_length,
        )

        if self.max_length <= MAX_OUTPUT_LENGTH:
            raise ValueError("max_length greater than MAX_OUTPUT_LENGTH")

        self.output_dims = (MAX_OUTPUT_LENGTH, 1)
        self.data_train = None
        self.data_val = None
        self.data_test = None

    @property
    def data_filename(self) -> Path:
        """Return name of dataset."""
        return (
            DATA_DIRNAME
            / f"ml_{self.max_length}_o{self.min_overlap:f}_{self.max_overlap:f}_ntr{self.num_train}_ntv{self.num_val}_nte{self.num_test}_{self.with_start_end_tokens}.h5"
        )

    def prepare_data(self) -> None:
        if self.data_filename.exists():
            return
        np.random.seed(SEED)
        self._generate_data("train")
        self._generate_data("val")
        self._generate_data("test")

    def setup(self, stage: str = None) -> None:
        logger.info("EMNISTLinesDataset loading data from HDF5...")
        if stage == "fit" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_train = f["x_train"][:]
                y_train = torch.LongTensor(f["y_train"][:])
                x_val = f["x_val"][:]
                y_val = torch.LongTensor(f["y_val"][:])

            self.data_train = BaseDataset(x_train, y_train, transform=_get_transform(augment=self.augment))
            self.data_val  = BaseDataset(x_val, y_val, transform=_get_transform(augment=self.augment))

        if stage == "test" or stage is None:
            with h5py.File(self.data_filename, "r") as f:
                x_test = f["x_test"][:]
                y_test = torch.LongTensor(f["y_test"][:])

            self.data_train = BaseDataset(x_test, y_test, transform=_get_transform(augment=False))

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
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data

    def _generate_data(self, split: str) -> None:
        logger.info(f"EMNISTLines generating data for {split}...")
        sentence_generator = SentenceGenerator(self.max_length - 2)  # Subtract by 2 because start/end token

        emnist = self.emnist
        emnist.prepare_data()
        emnist.setup()

        if split == "train":
            samples_by_char = _get_samples_by_char(emnist.x_train, emnist.y_train, emnist.mapping)
            num = self.num_train
        elif split == "val":
            samples_by_char = _get_samples_by_char(emnist.x_train, emnist.y_train, emnist.mapping)
            num = self.num_val
        elif split == "test":
            samples_by_char = _get_samples_by_char(emnist.x_test, emnist.y_test, emnist.mapping)
            num = self.num_test

        DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.data_filename, "w") as f:
            x, y = _create_dataset_of_images(
                    num, samples_by_char, sentence_generator, self.min_overlap, self.max_overlap, self.dims
                    )
            y = _convert_strings_to_labels(
                    y,
                    emnist.inverse_mapping,
                    length=MAX_OUTPUT_LENGTH
                    )
            f.create_dataset(f"x_{split}", data=x, dtype="u1", compression="lzf")
            f.create_dataset(f"y_{split}", data=y, dtype="u1", compression="lzf")

def _get_samples_by_char(samples: np.ndarray, labels: np.ndarray, mapping: Dict) -> defaultdict:
    samples_by_char = defaultdict(list)
    for sample, label in zip(samples, labels):
        samples_by_char[mapping[label]].append(sample)
    return samples_by_char


def _construct_image_from_string():
    pass


def _select_letter_samples_for_string(string: str, samples_by_char: defaultdict):
    pass


def _create_dataset_of_images(num_samples: int, samples_by_char: defaultdict, sentence_generator: SentenceGenerator, min_overlap: float, max_overlap: float, dims: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.zeros((num_samples, IMAGE_HEIGHT, dims[2]))
    labels = []
    for n in range(num_samples):
        label = sentence_generator.generate()
        crop = _construct_image_from_string()

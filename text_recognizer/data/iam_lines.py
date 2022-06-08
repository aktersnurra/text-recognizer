"""Class for IAM Lines dataset.

If not created, will generate a handwritten lines dataset from the IAM paragraphs
dataset.
"""
import json
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Type

from loguru import logger as log
import numpy as np
from PIL import Image, ImageFile, ImageOps
from torch import Tensor

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.base_dataset import (
    BaseDataset,
    convert_strings_to_labels,
    split_dataset,
)
from text_recognizer.data.iam import IAM
from text_recognizer.data.mappings import AbstractMapping, EmnistMapping
from text_recognizer.data.transforms.load_transform import load_transform_from_file
from text_recognizer.data.utils import image_utils


ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 4711
PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_lines"
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 1024
MAX_LABEL_LENGTH = 89
MAX_WORD_PIECE_LENGTH = 72


class IAMLines(BaseDataModule):
    """IAM handwritten lines dataset."""

    def __init__(
        self,
        mapping: Type[AbstractMapping],
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_fraction: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            mapping,
            transform,
            test_transform,
            target_transform,
            train_fraction,
            batch_size,
            num_workers,
            pin_memory,
        )
        self.dims = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
        self.output_dims = (MAX_LABEL_LENGTH, 1)

    def prepare_data(self) -> None:
        """Creates the IAM lines dataset if not existing."""
        if PROCESSED_DATA_DIRNAME.exists():
            return

        log.info("Cropping IAM lines regions...")
        iam = IAM(mapping=EmnistMapping())
        iam.prepare_data()
        crops_train, labels_train = line_crops_and_labels(iam, "train")
        crops_test, labels_test = line_crops_and_labels(iam, "test")

        shapes = np.array([crop.size for crop in crops_train + crops_test])
        aspect_ratios = shapes[:, 0] / shapes[:, 1]

        log.info("Saving images, labels, and statistics...")
        save_images_and_labels(
            crops_train, labels_train, "train", PROCESSED_DATA_DIRNAME
        )
        save_images_and_labels(crops_test, labels_test, "test", PROCESSED_DATA_DIRNAME)

        with (PROCESSED_DATA_DIRNAME / "_max_aspect_ratio.txt").open(mode="w") as f:
            f.write(str(aspect_ratios.max()))

    def setup(self, stage: str = None) -> None:
        """Load data for training/testing."""
        with (PROCESSED_DATA_DIRNAME / "_max_aspect_ratio.txt").open(mode="r") as f:
            max_aspect_ratio = float(f.read())
            image_width = int(IMAGE_HEIGHT * max_aspect_ratio)
            if image_width >= IMAGE_WIDTH:
                raise ValueError("image_width equal or greater than IMAGE_WIDTH")

        if stage == "fit" or stage is None:
            x_train, labels_train = load_line_crops_and_labels(
                "train", PROCESSED_DATA_DIRNAME
            )
            if self.output_dims[0] < max([len(labels) for labels in labels_train]) + 2:
                raise ValueError("Target length longer than max output length.")

            y_train = convert_strings_to_labels(
                labels_train, self.mapping.inverse_mapping, length=self.output_dims[0]
            )
            data_train = BaseDataset(
                x_train,
                y_train,
                transform=self.transform,
                target_transform=self.target_transform,
            )

            self.data_train, self.data_val = split_dataset(
                dataset=data_train, fraction=self.train_fraction, seed=SEED
            )

        if stage == "test" or stage is None:
            x_test, labels_test = load_line_crops_and_labels(
                "test", PROCESSED_DATA_DIRNAME
            )

            if self.output_dims[0] < max([len(labels) for labels in labels_test]) + 2:
                raise ValueError("Taget length longer than max output length.")

            y_test = convert_strings_to_labels(
                labels_test, self.mapping.inverse_mapping, length=self.output_dims[0]
            )
            self.data_test = BaseDataset(
                x_test,
                y_test,
                transform=self.test_transform,
                target_transform=self.target_transform,
            )

        if stage is None:
            self._verify_output_dims(labels_train, labels_test)

    def _verify_output_dims(self, labels_train: Tensor, labels_test: Tensor) -> None:
        max_label_length = max([len(label) for label in labels_train + labels_test]) + 2
        output_dims = (max_label_length, 1)
        if output_dims != self.output_dims:
            raise ValueError("Output dim does not match expected output dims.")

    def __repr__(self) -> str:
        """Return information about the dataset."""
        basic = (
            "IAM Lines dataset\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )

        if not any([self.data_train, self.data_val, self.data_test]):
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        x = x[0] if isinstance(x, list) else x
        xt = xt[0] if isinstance(xt, list) else xt
        data = (
            "Train/val/test sizes: "
            f"{len(self.data_train)}, "
            f"{len(self.data_val)}, "
            f"{len(self.data_test)}\n"
            "Train Batch x stats: "
            f"{(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            "Train Batch y stats: "
            f"{(y.shape, y.dtype, y.min(), y.max())}\n"
            "Test Batch x stats: "
            f"{(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            "Test Batch y stats: "
            f"{(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def line_crops_and_labels(iam: IAM, split: str) -> Tuple[List, List]:
    """Load IAM line labels and regions, and load image crops."""
    crops = []
    labels = []
    for filename in iam.form_filenames:
        if not iam.split_by_id[filename.stem] == split:
            continue
        image = image_utils.read_image_pil(filename)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)
        labels += iam.line_strings_by_id[filename.stem]
        crops += [
            image.crop([region[box] for box in ["x1", "y1", "x2", "y2"]])
            for region in iam.line_regions_by_id[filename.stem]
        ]
    if len(crops) != len(labels):
        raise ValueError("Length of crops does not match length of labels")
    return crops, labels


def save_images_and_labels(
    crops: Sequence[Image.Image], labels: Sequence[str], split: str, data_dirname: Path
) -> None:
    """Saves generated images and labels to disk."""
    (data_dirname / split).mkdir(parents=True, exist_ok=True)

    with (data_dirname / split / "_labels.json").open(mode="w") as f:
        json.dump(labels, f)

    for index, crop in enumerate(crops):
        crop.save(data_dirname / split / f"{index}.png")


def load_line_crops_and_labels(split: str, data_dirname: Path) -> Tuple[List, List]:
    """Load line crops and labels for given split from processed directoru."""
    with (data_dirname / split / "_labels.json").open(mode="r") as f:
        labels = json.load(f)

    crop_filenames = sorted(
        (data_dirname / split).glob("*.png"),
        key=lambda filename: int(Path(filename).stem),
    )
    crops = [
        image_utils.read_image_pil(filename, grayscale=True)
        for filename in crop_filenames
    ]

    if len(crops) != len(labels):
        raise ValueError("Length of crops does not match length of labels")

    return crops, labels


def generate_iam_lines() -> None:
    """Displays Iam Lines dataset statistics."""
    transform = load_transform_from_file("transform/lines.yaml")
    test_transform = load_transform_from_file("test_transform/lines.yaml")
    load_and_print_info(IAMLines(transform=transform, test_transform=test_transform))

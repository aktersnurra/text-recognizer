"""IAM Paragraphs Dataset class."""
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torchvision.transforms as T
from loguru import logger as log
from PIL import Image, ImageOps
from tqdm import tqdm

import text_recognizer.metadata.iam_paragraphs as metadata
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.base_dataset import (
    BaseDataset,
    convert_strings_to_labels,
    split_dataset,
)
from text_recognizer.data.iam import IAM
from text_recognizer.data.tokenizer import Tokenizer
from text_recognizer.data.transforms.pad import Pad
from text_recognizer.data.transforms.paragraph import ParagraphStem


class IAMParagraphs(BaseDataModule):
    """IAM handwriting database paragraphs."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_fraction: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            tokenizer,
            transform,
            test_transform,
            target_transform,
            train_fraction,
            batch_size,
            num_workers,
            pin_memory,
        )
        self.dims = (1, metadata.IMAGE_HEIGHT, metadata.IMAGE_WIDTH)
        self.output_dims = (metadata.MAX_LABEL_LENGTH, 1)

    def prepare_data(self) -> None:
        """Create data for training/testing."""
        if metadata.PROCESSED_DATA_DIRNAME.exists():
            return

        log.info("Cropping IAM paragraph regions and saving them along with labels...")

        iam = IAM(tokenizer=self.tokenizer)
        iam.prepare_data()

        properties = {}
        for split in ["train", "test"]:
            crops, labels = _get_paragraph_crops_and_labels(iam=iam, split=split)
            _save_crops_and_labels(crops=crops, labels=labels, split=split)

            properties.update(
                {
                    id_: {
                        "crop_shape": crops[id_].size[::-1],
                        "label_length": len(label),
                        "num_lines": _num_lines(label),
                    }
                    for id_, label in labels.items()
                }
            )

        with (metadata.PROCESSED_DATA_DIRNAME / "_properties.json").open("w") as f:
            json.dump(properties, f, indent=4)

    def setup(self, stage: str = None) -> None:
        """Loads the data for training/testing."""

        def _load_dataset(
            split: str, transform: T.Compose, target_transform: T.Compose
        ) -> BaseDataset:
            crops, labels = _load_processed_crops_and_labels(split)
            data = [resize_image(crop, metadata.IMAGE_SCALE_FACTOR) for crop in crops]
            targets = convert_strings_to_labels(
                strings=labels,
                mapping=self.tokenizer.inverse_mapping,
                length=self.output_dims[0],
            )
            return BaseDataset(
                data,
                targets,
                transform=transform,
                target_transform=target_transform,
            )

        log.info(f"Loading IAM paragraph regions and lines for {stage}...")
        _validate_data_dims(input_dims=self.dims, output_dims=self.output_dims)

        if stage == "fit" or stage is None:
            data_train = _load_dataset(
                split="train",
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.data_train, self.data_val = split_dataset(
                dataset=data_train, fraction=self.train_fraction, seed=metadata.SEED
            )

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(
                split="test",
                transform=self.test_transform,
                target_transform=self.target_transform,
            )

    def __repr__(self) -> str:
        """Return information about the dataset."""
        basic = (
            "IAM Paragraphs Dataset\n"
            f"Num classes: {len(self.tokenizer)}\n"
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


def get_dataset_properties() -> Dict:
    """Return properties describing the overall dataset."""
    with (metadata.PROCESSED_DATA_DIRNAME / "_properties.json").open("r") as f:
        properties = json.load(f)

    def _get_property_values(key: str) -> List:
        return [value[key] for value in properties.values()]

    crop_shapes = np.array(_get_property_values("crop_shape"))
    aspect_ratio = crop_shapes[:, 1] / crop_shapes[:, 0]
    return {
        "label_length": {
            "min": min(_get_property_values("label_length")),
            "max": max(_get_property_values("label_length")),
        },
        "num_lines": {
            "min": min(_get_property_values("num_lines")),
            "max": max(_get_property_values("num_lines")),
        },
        "crop_shape": {"min": crop_shapes.min(axis=0), "max": crop_shapes.max(axis=0)},
        "aspect_ratio": {
            "min": aspect_ratio.min(axis=0),
            "max": aspect_ratio.max(axis=0),
        },
    }


def _validate_data_dims(
    input_dims: Optional[Tuple[int, ...]], output_dims: Optional[Tuple[int, ...]]
) -> None:
    """Validates input and output dimensions against the properties of the dataset."""
    properties = get_dataset_properties()

    max_image_shape = properties["crop_shape"]["max"] / metadata.IMAGE_SCALE_FACTOR
    if (
        input_dims is not None
        and input_dims[1] < max_image_shape[0]
        and input_dims[2] < max_image_shape[1]
    ):
        raise ValueError(f"{input_dims} less than {max_image_shape}")

    if (
        output_dims is not None
        and output_dims[0] < properties["label_length"]["max"] + 2
    ):
        raise ValueError(
            f"{output_dims} less than {properties['label_length']['max'] + 2}"
        )


def resize_image(image: Image.Image, scale_factor: int) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize(
        (image.width // scale_factor, image.height // scale_factor),
        resample=Image.BILINEAR,
    )


def _get_paragraph_crops_and_labels(
    iam: IAM, split: str
) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """Load IAM paragraph crops and labels for a given set."""
    crops = {}
    labels = {}
    for form_filename in tqdm(
        iam.form_filenames, desc=f"Processing {split} paragraphs"
    ):
        id_ = form_filename.stem
        if not iam.split_by_id[id_] == split:
            continue
        image = Image.open(form_filename)
        image = ImageOps.grayscale(image)
        image = ImageOps.invert(image)

        line_regions = iam.line_regions_by_id[id_]
        paragraph_box = [
            min([region["x1"] for region in line_regions]),
            min([region["y1"] for region in line_regions]),
            max([region["x2"] for region in line_regions]),
            max([region["y2"] for region in line_regions]),
        ]
        lines = iam.line_strings_by_id[id_]

        crops[id_] = image.crop(paragraph_box)
        labels[id_] = metadata.NEW_LINE_TOKEN.join(lines)

    if len(crops) != len(labels):
        raise ValueError(f"Crops ({len(crops)}) does not match labels ({len(labels)})")

    return crops, labels


def _save_crops_and_labels(
    crops: Dict[str, Image.Image], labels: Dict[str, str], split: str
) -> None:
    """Save crops, labels, and shapes of crops of a split."""
    (metadata.PROCESSED_DATA_DIRNAME / split).mkdir(parents=True, exist_ok=True)

    with _labels_filename(split).open("w") as f:
        json.dump(labels, f, indent=4)

    for id_, crop in crops.items():
        crop.save(_crop_filename(id_, split))


def _load_processed_crops_and_labels(
    split: str,
) -> Tuple[Sequence[Image.Image], Sequence[str]]:
    """Load processed crops and labels for given split."""
    with _labels_filename(split).open("r") as f:
        labels = json.load(f)

    sorted_ids = sorted(labels.keys())
    ordered_crops = [
        Image.open(_crop_filename(id_, split)).convert("L") for id_ in sorted_ids
    ]
    ordered_labels = [labels[id_] for id_ in sorted_ids]

    if len(ordered_crops) != len(ordered_labels):
        raise ValueError(
            f"Crops ({len(ordered_crops)}) does not match labels ({len(ordered_labels)})"
        )
    return ordered_crops, ordered_labels


def _labels_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return metadata.PROCESSED_DATA_DIRNAME / split / "_labels.json"


def _crop_filename(id: str, split: str) -> Path:
    """Return filename of processed crop."""
    return metadata.PROCESSED_DATA_DIRNAME / split / f"{id}.png"


def _num_lines(label: str) -> int:
    """Return the number of lines of text in label."""
    return label.count("\n") + 1


def create_iam_paragraphs() -> None:
    """Loads and displays dataset statistics."""
    transform = ParagraphStem()
    test_transform = ParagraphStem()
    target_transform = Pad(metadata.MAX_LABEL_LENGTH, 3)
    load_and_print_info(
        IAMParagraphs(
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
        )
    )

"""IAM Paragraphs Dataset class."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import attr
from loguru import logger as log
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from text_recognizer.data.base_dataset import (
    BaseDataset,
    convert_strings_to_labels,
    split_dataset,
)
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.emnist_mapping import EmnistMapping
from text_recognizer.data.iam import IAM
from text_recognizer.data.transforms import WordPiece


PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "iam_paragraphs"

NEW_LINE_TOKEN = "\n"

SEED = 4711
IMAGE_SCALE_FACTOR = 2
IMAGE_HEIGHT = 1152 // IMAGE_SCALE_FACTOR
IMAGE_WIDTH = 1280 // IMAGE_SCALE_FACTOR
MAX_LABEL_LENGTH = 682


@attr.s(auto_attribs=True, repr=False)
class IAMParagraphs(BaseDataModule):
    """IAM handwriting database paragraphs."""

    word_pieces: bool = attr.ib(default=False)
    augment: bool = attr.ib(default=True)
    train_fraction: float = attr.ib(default=0.8)
    dims: Tuple[int, int, int] = attr.ib(
        init=False, default=(1, IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    output_dims: Tuple[int, int] = attr.ib(init=False, default=(MAX_LABEL_LENGTH, 1))

    def prepare_data(self) -> None:
        """Create data for training/testing."""
        if PROCESSED_DATA_DIRNAME.exists():
            return

        log.info("Cropping IAM paragraph regions and saving them along with labels...")

        iam = IAM(mapping=EmnistMapping(extra_symbols={NEW_LINE_TOKEN,}))
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

        with (PROCESSED_DATA_DIRNAME / "_properties.json").open("w") as f:
            json.dump(properties, f, indent=4)

    def setup(self, stage: str = None) -> None:
        """Loads the data for training/testing."""

        def _load_dataset(split: str, augment: bool) -> BaseDataset:
            crops, labels = _load_processed_crops_and_labels(split)
            data = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops]
            targets = convert_strings_to_labels(
                strings=labels,
                mapping=self.mapping.inverse_mapping,
                length=self.output_dims[0],
            )
            return BaseDataset(
                data,
                targets,
                transform=get_transform(image_shape=self.dims[1:], augment=augment),
                target_transform=get_target_transform(self.word_pieces),
            )

        log.info(f"Loading IAM paragraph regions and lines for {stage}...")
        _validate_data_dims(input_dims=self.dims, output_dims=self.output_dims)

        if stage == "fit" or stage is None:
            data_train = _load_dataset(split="train", augment=self.augment)
            self.data_train, self.data_val = split_dataset(
                dataset=data_train, fraction=self.train_fraction, seed=SEED
            )

        if stage == "test" or stage is None:
            self.data_test = _load_dataset(split="test", augment=False)

    def __repr__(self) -> str:
        """Return information about the dataset."""
        basic = (
            "IAM Paragraphs Dataset\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )

        if not any([self.data_train, self.data_val, self.data_test]):
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            "Train/val/test sizes: "
            f"{len(self.data_train)}, "
            f"{len(self.data_val)}, "
            f"{len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def get_dataset_properties() -> Dict:
    """Return properties describing the overall dataset."""
    with (PROCESSED_DATA_DIRNAME / "_properties.json").open("r") as f:
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
        "crop_shape": {"min": crop_shapes.min(axis=0), "max": crop_shapes.max(axis=0),},
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

    max_image_shape = properties["crop_shape"]["max"] / IMAGE_SCALE_FACTOR
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
        labels[id_] = NEW_LINE_TOKEN.join(lines)

    if len(crops) != len(labels):
        raise ValueError(f"Crops ({len(crops)}) does not match labels ({len(labels)})")

    return crops, labels


def _save_crops_and_labels(
    crops: Dict[str, Image.Image], labels: Dict[str, str], split: str
) -> None:
    """Save crops, labels, and shapes of crops of a split."""
    (PROCESSED_DATA_DIRNAME / split).mkdir(parents=True, exist_ok=True)

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


def get_transform(image_shape: Tuple[int, int], augment: bool) -> T.Compose:
    """Get transformations for images."""
    if augment:
        transforms_list = [
            T.RandomCrop(
                size=image_shape,
                padding=None,
                pad_if_needed=True,
                fill=0,
                padding_mode="constant",
            ),
            T.ColorJitter(brightness=(0.8, 1.6)),
            T.RandomAffine(
                degrees=1, shear=(-10, 10), interpolation=InterpolationMode.BILINEAR,
            ),
        ]
    else:
        transforms_list = [T.CenterCrop(image_shape)]
    transforms_list.append(T.ToTensor())
    return T.Compose(transforms_list)


def get_target_transform(word_pieces: bool) -> Optional[T.Compose]:
    """Transform emnist characters to word pieces."""
    return T.Compose([WordPiece()]) if word_pieces else None


def _labels_filename(split: str) -> Path:
    """Return filename of processed labels."""
    return PROCESSED_DATA_DIRNAME / split / "_labels.json"


def _crop_filename(id: str, split: str) -> Path:
    """Return filename of processed crop."""
    return PROCESSED_DATA_DIRNAME / split / f"{id}.png"


def _num_lines(label: str) -> int:
    """Return the number of lines of text in label."""
    return label.count("\n") + 1


def create_iam_paragraphs() -> None:
    load_and_print_info(IAMParagraphs)

"""IAM Synthetic Paragraphs Dataset class."""
import random
from typing import Any, List, Sequence, Tuple

import attr
from loguru import logger
import numpy as np
from PIL import Image

from text_recognizer.data.base_dataset import (
    BaseDataset,
    convert_strings_to_labels,
)
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.iam_paragraphs import (
    get_dataset_properties,
    get_transform,
    get_target_transform,
    NEW_LINE_TOKEN,
    IAMParagraphs,
    IMAGE_SCALE_FACTOR,
    resize_image,
)
from text_recognizer.data.iam import IAM
from text_recognizer.data.iam_lines import (
    line_crops_and_labels,
    load_line_crops_and_labels,
    save_images_and_labels,
)


PROCESSED_DATA_DIRNAME = (
    BaseDataModule.data_dirname() / "processed" / "iam_synthetic_paragraphs"
)


@attr.s(auto_attribs=True, repr=False)
class IAMSyntheticParagraphs(IAMParagraphs):
    """IAM Handwriting database of synthetic paragraphs."""

    def prepare_data(self) -> None:
        """Prepare IAM lines to be used to generate paragraphs."""
        if PROCESSED_DATA_DIRNAME.exists():
            return

        logger.info("Preparing IAM lines for synthetic paragraphs dataset.")
        logger.info("Cropping IAM line regions and loading labels.")

        iam = IAM()
        iam.prepare_data()

        crops_train, labels_train = line_crops_and_labels(iam, "train")
        crops_test, labels_test = line_crops_and_labels(iam, "test")

        crops_train = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops_train]
        crops_test = [resize_image(crop, IMAGE_SCALE_FACTOR) for crop in crops_test]

        logger.info(f"Saving images and labels at {PROCESSED_DATA_DIRNAME}")
        save_images_and_labels(
            crops_train, labels_train, "train", PROCESSED_DATA_DIRNAME
        )
        save_images_and_labels(crops_test, labels_test, "test", PROCESSED_DATA_DIRNAME)

    def setup(self, stage: str = None) -> None:
        """Loading synthetic dataset."""

        logger.info(f"IAM Synthetic dataset steup for stage {stage}...")

        if stage == "fit" or stage is None:
            line_crops, line_labels = load_line_crops_and_labels(
                "train", PROCESSED_DATA_DIRNAME
            )
            data, paragraphs_labels = generate_synthetic_paragraphs(
                line_crops=line_crops, line_labels=line_labels
            )

            targets = convert_strings_to_labels(
                strings=paragraphs_labels,
                mapping=self.inverse_mapping,
                length=self.output_dims[0],
            )
            self.data_train = BaseDataset(
                data,
                targets,
                transform=get_transform(
                    image_shape=self.dims[1:], augment=self.augment
                ),
                target_transform=get_target_transform(self.word_pieces),
            )

    def __repr__(self) -> str:
        """Return information about the dataset."""
        basic = (
            "IAM Synthetic Paragraphs Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.mapping)}\n"
            f"Input dims : {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, 0, 0\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data


def generate_synthetic_paragraphs(
    line_crops: List[Image.Image], line_labels: List[str], max_batch_size: int = 9
) -> Tuple[List[Image.Image], List[str]]:
    """Generate synthetic paragraphs from randomly joining different subsets."""
    paragraphs_properties = get_dataset_properties()

    indices = list(range(len(line_labels)))

    if max_batch_size >= paragraphs_properties["num_lines"]["max"]:
        raise ValueError("max_batch_size greater or equalt to max num lines.")

    batched_indices_list = [[index] for index in indices]
    batched_indices_list.extend(
        generate_random_batches(
            values=indices, min_batch_size=2, max_batch_size=max_batch_size // 2
        )
    )
    batched_indices_list.extend(
        generate_random_batches(
            values=indices, min_batch_size=2, max_batch_size=max_batch_size
        )
    )
    batched_indices_list.extend(
        generate_random_batches(
            values=indices,
            min_batch_size=max_batch_size // 2 + 1,
            max_batch_size=max_batch_size,
        )
    )

    paragraphs_crops, paragraphs_labels = [], []
    for paragraph_indices in batched_indices_list:
        paragraph_label = NEW_LINE_TOKEN.join(
            [line_labels[i] for i in paragraph_indices]
        )
        if len(paragraph_label) > paragraphs_properties["label_length"]["max"]:
            logger.info(
                "Label longer than longest label in original IAM paragraph dataset - hence dropping."
            )
            continue

        paragraph_crop = join_line_crops_to_form_paragraph(
            [line_crops[i] for i in paragraph_indices]
        )
        max_paragraph_shape = paragraphs_properties["crop_shape"]["max"]

        if (
            paragraph_crop.height > max_paragraph_shape[0]
            or paragraph_crop.width > max_paragraph_shape[1]
        ):
            logger.info(
                "Crop larger than largest crop in original IAM paragraphs dataset - hence dropping"
            )
            continue

        paragraphs_crops.append(paragraph_crop)
        paragraphs_labels.append(paragraph_label)

    if len(paragraphs_crops) != len(paragraphs_labels):
        raise ValueError("Number of crops does not match number of labels.")

    return paragraphs_crops, paragraphs_labels


def join_line_crops_to_form_paragraph(line_crops: Sequence[Image.Image]) -> Image.Image:
    """Horizontally stack line crops and return a single image forming a paragraph."""
    crop_shapes = np.array([line.size[::-1] for line in line_crops])
    paragraph_height = crop_shapes[:, 0].sum()
    paragraph_width = crop_shapes[:, 1].max()

    paragraph_image = Image.new(
        mode="L", size=(paragraph_width, paragraph_height), color=0
    )
    current_height = 0
    for line_crop in line_crops:
        paragraph_image.paste(line_crop, box=(0, current_height))
        current_height += line_crop.height

    return paragraph_image


def generate_random_batches(
    values: List[Any], min_batch_size: int, max_batch_size: int
) -> List[List[Any]]:
    """Generate random batches of elements in values without replacement."""
    shuffled_values = values.copy()
    random.shuffle(shuffled_values)

    start_index = 0
    grouped_values_list = []
    while start_index < len(shuffled_values):
        num_values = random.randint(min_batch_size, max_batch_size)
        grouped_values_list.append(
            shuffled_values[start_index : start_index + num_values]
        )
        start_index += num_values

    if sum([len(grp) for grp in grouped_values_list]) != len(values):
        raise ValueError("Length of groups does not match length of values.")

    return grouped_values_list


def create_synthetic_iam_paragraphs() -> None:
    load_and_print_info(IAMSyntheticParagraphs)

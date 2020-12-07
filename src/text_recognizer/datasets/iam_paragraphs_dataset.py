"""IamParagraphsDataset class and functions for data processing."""
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import click
import cv2
import h5py
from loguru import logger
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import ToTensor

from text_recognizer import util
from text_recognizer.datasets.dataset import Dataset
from text_recognizer.datasets.iam_dataset import IamDataset
from text_recognizer.datasets.util import (
    compute_sha256,
    DATA_DIRNAME,
    download_url,
    EmnistMapper,
)

INTERIM_DATA_DIRNAME = DATA_DIRNAME / "interim" / "iam_paragraphs"
DEBUG_CROPS_DIRNAME = INTERIM_DATA_DIRNAME / "debug_crops"
PROCESSED_DATA_DIRNAME = DATA_DIRNAME / "processed" / "iam_paragraphs"
CROPS_DIRNAME = PROCESSED_DATA_DIRNAME / "crops"
GT_DIRNAME = PROCESSED_DATA_DIRNAME / "gt"

PARAGRAPH_BUFFER = 50  # Pixels in the IAM form images to leave around the lines.
TEST_FRACTION = 0.2
SEED = 4711


class IamParagraphsDataset(Dataset):
    """IAM Paragraphs dataset for paragraphs of handwritten text."""

    def __init__(
        self,
        train: bool = False,
        subsample_fraction: float = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            train=train,
            subsample_fraction=subsample_fraction,
            transform=transform,
            target_transform=target_transform,
        )
        # Load Iam dataset.
        self.iam_dataset = IamDataset()

        self.num_classes = 3
        self._input_shape = (256, 256)
        self._output_shape = self._input_shape + (self.num_classes,)
        self._ids = None

    def __getitem__(self, index: Union[Tensor, int]) -> Tuple[Tensor, Tensor]:
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

        seed = np.random.randint(SEED)
        random.seed(seed)  # apply this seed to target tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transform:
            data = self.transform(data)

        random.seed(seed)  # apply this seed to target tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.target_transform:
            targets = self.target_transform(targets)

        return data, targets.long()

    @property
    def ids(self) -> Tensor:
        """Ids of the dataset."""
        return self._ids

    def get_data_and_target_from_id(self, id_: str) -> Tuple[Tensor, Tensor]:
        """Get data target pair from id."""
        ind = self.ids.index(id_)
        return self.data[ind], self.targets[ind]

    def load_or_generate_data(self) -> None:
        """Load or generate dataset data."""
        num_actual = len(list(CROPS_DIRNAME.glob("*.jpg")))
        num_targets = len(self.iam_dataset.line_regions_by_id)

        if num_actual < num_targets - 2:
            self._process_iam_paragraphs()

        self._data, self._targets, self._ids = _load_iam_paragraphs()
        self._get_random_split()
        self._subsample()

    def _get_random_split(self) -> None:
        np.random.seed(SEED)
        num_train = int((1 - TEST_FRACTION) * self.data.shape[0])
        indices = np.random.permutation(self.data.shape[0])
        train_indices, test_indices = indices[:num_train], indices[num_train:]
        if self.train:
            self._data = self.data[train_indices]
            self._targets = self.targets[train_indices]
        else:
            self._data = self.data[test_indices]
            self._targets = self.targets[test_indices]

    def _process_iam_paragraphs(self) -> None:
        """Crop the part with the text.

        For each page, crop out the part of it that correspond to the paragraph of text, and make sure all crops are
        self.input_shape. The ground truth data is the same size, with a one-hot vector at each pixel
        corresponding to labels 0=background, 1=odd-numbered line, 2=even-numbered line
        """
        crop_dims = self._decide_on_crop_dims()
        CROPS_DIRNAME.mkdir(parents=True, exist_ok=True)
        DEBUG_CROPS_DIRNAME.mkdir(parents=True, exist_ok=True)
        GT_DIRNAME.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Cropping paragraphs, generating ground truth, and saving debugging images to {DEBUG_CROPS_DIRNAME}"
        )
        for filename in self.iam_dataset.form_filenames:
            id_ = filename.stem
            line_region = self.iam_dataset.line_regions_by_id[id_]
            _crop_paragraph_image(filename, line_region, crop_dims, self.input_shape)

    def _decide_on_crop_dims(self) -> Tuple[int, int]:
        """Decide on the dimensions to crop out of the form image.

        Since image width is larger than a comfortable crop around the longest paragraph,
        we will make the crop a square form factor.
        And since the found dimensions 610x610 are pretty close to 512x512,
        we might as well resize crops and make it exactly that, which lets us
        do all kinds of power-of-2 pooling and upsampling should we choose to.

        Returns:
            Tuple[int, int]: A tuple of crop dimensions.

        Raises:
            RuntimeError: When max crop height is larger than max crop width.

        """

        sample_form_filename = self.iam_dataset.form_filenames[0]
        sample_image = util.read_image(sample_form_filename, grayscale=True)
        max_crop_width = sample_image.shape[1]
        max_crop_height = _get_max_paragraph_crop_height(
            self.iam_dataset.line_regions_by_id
        )
        if not max_crop_height <= max_crop_width:
            raise RuntimeError(
                f"Max crop height is larger then max crop width: {max_crop_height} >= {max_crop_width}"
            )

        crop_dims = (max_crop_width, max_crop_width)
        logger.info(
            f"Max crop width and height were found to be {max_crop_width}x{max_crop_height}."
        )
        logger.info(f"Setting them to {max_crop_width}x{max_crop_width}")
        return crop_dims

    def __repr__(self) -> str:
        """Return info about the dataset."""
        return (
            "IAM Paragraph Dataset\n"  # pylint: disable=no-member
            f"Num classes: {self.num_classes}\n"
            f"Data: {self.data.shape}\n"
            f"Targets: {self.targets.shape}\n"
        )


def _get_max_paragraph_crop_height(line_regions_by_id: Dict) -> int:
    heights = []
    for regions in line_regions_by_id.values():
        min_y1 = min(region["y1"] for region in regions) - PARAGRAPH_BUFFER
        max_y2 = max(region["y2"] for region in regions) + PARAGRAPH_BUFFER
        height = max_y2 - min_y1
        heights.append(height)
    return max(heights)


def _crop_paragraph_image(
    filename: str, line_regions: Dict, crop_dims: Tuple[int, int], final_dims: Tuple
) -> None:
    image = util.read_image(filename, grayscale=True)

    min_y1 = min(region["y1"] for region in line_regions) - PARAGRAPH_BUFFER
    max_y2 = max(region["y2"] for region in line_regions) + PARAGRAPH_BUFFER
    height = max_y2 - min_y1
    crop_height = crop_dims[0]
    buffer = (crop_height - height) // 2

    # Generate image crop.
    image_crop = 255 * np.ones(crop_dims, dtype=np.uint8)
    try:
        image_crop[buffer : buffer + height] = image[min_y1:max_y2]
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Rescued {filename}: {e}")
        return

    # Generate ground truth.
    gt_image = np.zeros_like(image_crop, dtype=np.uint8)
    for index, region in enumerate(line_regions):
        gt_image[
            (region["y1"] - min_y1 + buffer) : (region["y2"] - min_y1 + buffer),
            region["x1"] : region["x2"],
        ] = (index % 2 + 1)

    # Generate image for debugging.
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("Set1")
    image_crop_for_debug = np.dstack([image_crop, image_crop, image_crop])
    for index, region in enumerate(line_regions):
        color = [255 * _ for _ in cmap(index)[:-1]]
        cv2.rectangle(
            image_crop_for_debug,
            (region["x1"], region["y1"] - min_y1 + buffer),
            (region["x2"], region["y2"] - min_y1 + buffer),
            color,
            3,
        )
    image_crop_for_debug = cv2.resize(
        image_crop_for_debug, final_dims, interpolation=cv2.INTER_AREA
    )
    util.write_image(image_crop_for_debug, DEBUG_CROPS_DIRNAME / f"{filename.stem}.jpg")

    image_crop = cv2.resize(image_crop, final_dims, interpolation=cv2.INTER_AREA)
    util.write_image(image_crop, CROPS_DIRNAME / f"{filename.stem}.jpg")

    gt_image = cv2.resize(gt_image, final_dims, interpolation=cv2.INTER_NEAREST)
    util.write_image(gt_image, GT_DIRNAME / f"{filename.stem}.png")


def _load_iam_paragraphs() -> None:
    logger.info("Loading IAM paragraph crops and ground truth from image files...")
    images = []
    gt_images = []
    ids = []
    for filename in CROPS_DIRNAME.glob("*.jpg"):
        id_ = filename.stem
        image = util.read_image(filename, grayscale=True)
        image = 1.0 - image / 255

        gt_filename = GT_DIRNAME / f"{id_}.png"
        gt_image = util.read_image(gt_filename, grayscale=True)

        images.append(image)
        gt_images.append(gt_image)
        ids.append(id_)
    images = np.array(images).astype(np.float32)
    gt_images = np.array(gt_images).astype(np.uint8)
    ids = np.array(ids)
    return images, gt_images, ids


@click.command()
@click.option(
    "--subsample_fraction",
    type=float,
    default=None,
    help="The subsampling factor of the dataset.",
)
def main(subsample_fraction: float) -> None:
    """Load dataset and print info."""
    logger.info("Creating train set...")
    dataset = IamParagraphsDataset(train=True, subsample_fraction=subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)
    logger.info("Creating test set...")
    dataset = IamParagraphsDataset(subsample_fraction=subsample_fraction)
    dataset.load_or_generate_data()
    print(dataset)


if __name__ == "__main__":
    main()

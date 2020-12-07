"""Full model.

Takes an image and returns the text in the image, by first segmenting the image with a LineDetector, then extracting the
each crop of the image corresponding to line regions, and feeding them to a LinePredictor model that outputs the text
in each region.
"""
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch

from text_recognizer.models import SegmentationModel, TransformerModel
from text_recognizer.util import read_image


class ParagraphTextRecognizor:
    """Given an image of a single handwritten character, recognizes it."""

    def __init__(self, line_predictor_args: Dict, line_detector_args: Dict) -> None:
        self._line_predictor = TransformerModel(**line_predictor_args)
        self._line_detector = SegmentationModel(**line_detector_args)
        self._line_detector.eval()
        self._line_predictor.eval()

    def predict(self, image_or_filename: Union[str, np.ndarray]) -> Tuple:
        """Takes an image and returns all text within it."""
        image = (
            read_image(image_or_filename)
            if isinstance(image_or_filename, str)
            else image_or_filename
        )

        line_region_crops = self._get_line_region_crops(image)
        processed_line_region_crops = [
            self._process_image_for_line_predictor(image=crop)
            for crop in line_region_crops
        ]
        line_region_strings = [
            self.line_predictor_model.predict_on_image(crop)[0]
            for crop in processed_line_region_crops
        ]

        return " ".join(line_region_strings), line_region_crops

    def _get_line_region_crops(
        self, image: np.ndarray, min_crop_len_factor: float = 0.02
    ) -> List[np.ndarray]:
        """Returns all the crops of text lines in a square image."""
        processed_image, scale_down_factor = self._process_image_for_line_detector(
            image
        )
        line_segmentation = self._line_detector.predict_on_image(processed_image)
        bounding_boxes = _find_line_bounding_boxes(line_segmentation)

        bounding_boxes = (bounding_boxes * scale_down_factor).astype(int)

        min_crop_len = int(min_crop_len_factor * min(image.shape[0], image.shape[1]))
        line_region_crops = [
            image[y : y + h, x : x + w]
            for x, y, w, h in bounding_boxes
            if w >= min_crop_len and h >= min_crop_len
        ]
        return line_region_crops

    def _process_image_for_line_detector(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Convert uint8 image to float image with black background with shape self._line_detector.image_shape."""
        resized_image, scale_down_factor = _resize_image_for_line_detector(
            image=image, max_shape=self._line_detector.image_shape
        )
        resized_image = (1.0 - resized_image / 255).astype("float32")
        return resized_image, scale_down_factor

    def _process_image_for_line_predictor(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing of image before feeding it to the LinePrediction model.

        Convert uint8 image to float image with black background with shape
        self._line_predictor.image_shape while maintaining the image aspect ratio.

        Args:
            image (np.ndarray): Crop of text line.

        Returns:
            np.ndarray: Processed crop for feeding line predictor.
        """
        expected_shape = self._line_detector.image_shape
        scale_factor = (np.array(expected_shape) / np.array(image.shape)).min()
        scaled_image = cv2.resize(
            image,
            dsize=None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_AREA,
        )

        pad_with = (
            (0, expected_shape[0] - scaled_image.shape[0]),
            (0, expected_shape[1] - scaled_image.shape[1]),
        )

        padded_image = np.pad(
            scaled_image, pad_with=pad_with, mode="constant", constant_values=255
        )
        return 1 - padded_image / 255


def _find_line_bounding_boxes(line_segmentation: np.ndarray) -> np.ndarray:
    """Given a line segmentation, find bounding boxes for connected-component regions corresponding to non-0 labels."""

    def _find_line_bounding_boxes_in_channel(
        line_segmentation_channel: np.ndarray,
    ) -> np.ndarray:
        line_segmentation_image = cv2.dilate(
            line_segmentation_channel, kernel=np.ones((3, 3)), iterations=1
        )
        line_activation_image = (line_segmentation_image * 255).astype("uint8")
        line_activation_image = cv2.threshold(
            line_activation_image, 0.5, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]

        bounding_cnts, _ = cv2.findContours(
            line_segmentation_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return np.array([cv2.boundingRect(cnt) for cnt in bounding_cnts])

    bounding_boxes = np.concatenate(
        [
            _find_line_bounding_boxes_in_channel(line_segmentation[:, :, i])
            for i in [1, 2]
        ],
        axis=0,
    )

    return bounding_boxes[np.argsort(bounding_boxes[:, 1])]


def _resize_image_for_line_detector(
    image: np.ndarray, max_shape: Tuple[int, int]
) -> Tuple[np.ndarray, float]:
    """Resize the image to less than the max_shape while maintaining the aspect ratio."""
    scale_down_factor = max(np.ndarray(image.shape) / np.ndarray(max_shape))
    if scale_down_factor == 1:
        return image.copy(), scale_down_factor
    resize_image = cv2.resize(
        image,
        dsize=None,
        fx=1 / scale_down_factor,
        fy=1 / scale_down_factor,
        interpolation=cv2.INTER_AREA,
    )
    return resize_image, scale_down_factor

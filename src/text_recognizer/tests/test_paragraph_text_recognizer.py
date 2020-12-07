"""Test for ParagraphTextRecognizer class."""
import os
from pathlib import Path
import unittest

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizor
import text_recognizer.util as util


SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support" / "iam_paragraph"

# Prevent using GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestParagraphTextRecognizor(unittest.TestCase):
    """Test that it can take non-square images of max dimension larger than 256px."""

    def test_filename(self) -> None:
        """Test model on support image."""
        line_predictor_args = {
            "dataset": "EmnistLineDataset",
            "network_fn": "CNNTransformer",
        }
        line_detector_args = {"dataset": "EmnistLineDataset", "network_fn": "UNet"}
        model = ParagraphTextRecognizor(
            line_predictor_args=line_predictor_args,
            line_detector_args=line_detector_args,
        )
        num_text_lines_by_name = {"a01-000u-cropped": 7}
        for filename in (SUPPORT_DIRNAME).glob("*.jpg"):
            full_image = util.read_image(str(filename), grayscale=True)
            predicted_text, line_region_crops = model.predict(full_image)
            print(predicted_text)
            self.assertTrue(
                len(line_region_crops), num_text_lines_by_name[filename.stem]
            )

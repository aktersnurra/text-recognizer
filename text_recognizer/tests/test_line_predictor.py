"""Tests for LinePredictor."""
import os
from pathlib import Path
import unittest


import editdistance
import numpy as np

from text_recognizer.datasets import IamLinesDataset
from text_recognizer.line_predictor import LinePredictor
import text_recognizer.util as util

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support"

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestEmnistLinePredictor(unittest.TestCase):
    """Test LinePredictor class on the EmnistLines dataset."""

    def test_filename(self) -> None:
        """Test that LinePredictor correctly predicts on single images, for several test images."""
        predictor = LinePredictor(
            dataset="EmnistLineDataset", network_fn="CNNTransformer"
        )

        for filename in (SUPPORT_DIRNAME / "emnist_lines").glob("*.png"):
            pred, conf = predictor.predict(str(filename))
            true = str(filename.stem)
            edit_distance = editdistance.eval(pred, true) / len(pred)
            print(
                f'Pred: "{pred}" | Confidence: {conf} | True: {true} | Edit distance: {edit_distance}'
            )
            self.assertLess(edit_distance, 0.2)

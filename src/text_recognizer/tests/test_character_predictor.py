"""Test for CharacterPredictor class."""
import importlib
import os
from pathlib import Path
import unittest

from loguru import logger

from text_recognizer.character_predictor import CharacterPredictor
from text_recognizer.networks import MLP

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support" / "emnist"

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestCharacterPredictor(unittest.TestCase):
    """Tests for the CharacterPredictor class."""

    def test_filename(self) -> None:
        """Test that CharacterPredictor correctly predicts on a single image, for serveral test images."""
        network_fn_ = MLP
        predictor = CharacterPredictor(network_fn=network_fn_)

        for filename in SUPPORT_DIRNAME.glob("*.png"):
            pred, conf = predictor.predict(str(filename))
            logger.info(
                f"Prediction: {pred} at confidence: {conf} for image with character {filename.stem}"
            )
            self.assertEqual(pred, filename.stem)
            self.assertGreater(conf, 0.7)

"""Test for CharacterPredictor class."""
import importlib
import os
from pathlib import Path
import unittest

import click
from loguru import logger

from text_recognizer.character_predictor import CharacterPredictor
from text_recognizer.networks import MLP

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support" / "emnist"

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestCharacterPredictor(unittest.TestCase):
    """Tests for the CharacterPredictor class."""

    # @click.command()
    # @click.option(
    #     "--network", type=str, help="Network to load, e.g. MLP or LeNet.", default="MLP"
    # )
    def test_filename(self) -> None:
        """Test that CharacterPredictor correctly predicts on a single image, for serveral test images."""
        network_module = importlib.import_module("text_recognizer.networks")
        network_fn_ = getattr(network_module, "MLP")
        # network_args = {"input_size": [28, 28], "output_size": 62, "dropout_rate": 0}
        network_args = {"input_size": 784, "output_size": 62, "dropout_rate": 0.2}
        predictor = CharacterPredictor(
            network_fn=network_fn_, network_args=network_args
        )

        for filename in SUPPORT_DIRNAME.glob("*.png"):
            pred, conf = predictor.predict(str(filename))
            logger.info(
                f"Prediction: {pred} at confidence: {conf} for image with character {filename.stem}"
            )
            self.assertEqual(pred, filename.stem)
            self.assertGreater(conf, 0.7)

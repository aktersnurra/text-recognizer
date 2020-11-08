"""LinePredictor class."""
import importlib
from typing import Tuple, Union

import numpy as np
from torch import nn

from text_recognizer import datasets, networks
from text_recognizer.models import VisionTransformerModel
from text_recognizer.util import read_image


class LinePredictor:
    """Given an image of a line of handwritten text, recognizes the text content."""

    def __init__(self, dataset: str, network_fn: str) -> None:
        network_fn = getattr(networks, network_fn)
        dataset = getattr(datasets, dataset)
        self.model = VisionTransformerModel(network_fn=network_fn, dataset=dataset)
        self.model.eval()

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single images contianing a handwritten character."""
        if isinstance(image_or_filename, str):
            image = read_image(image_or_filename, grayscale=True)
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)

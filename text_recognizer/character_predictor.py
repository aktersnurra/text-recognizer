"""CharacterPredictor class."""
from typing import Dict, Tuple, Type, Union

import numpy as np
from torch import nn

from text_recognizer import datasets, networks
from text_recognizer.models import CharacterModel
from text_recognizer.util import read_image


class CharacterPredictor:
    """Recognizes the character in handwritten character images."""

    def __init__(self, network_fn: str, dataset: str) -> None:
        """Intializes the CharacterModel and load the pretrained weights."""
        network_fn = getattr(networks, network_fn)
        dataset = getattr(datasets, dataset)
        self.model = CharacterModel(network_fn=network_fn, dataset=dataset)
        self.model.eval()
        self.model.use_swa_model()

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single images contianing a handwritten character."""
        if isinstance(image_or_filename, str):
            image = read_image(image_or_filename, grayscale=True)
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)

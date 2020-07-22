"""CharacterPredictor class."""
from typing import Dict, Tuple, Type, Union

import numpy as np
from torch import nn

from text_recognizer.models import CharacterModel
from text_recognizer.util import read_image


class CharacterPredictor:
    """Recognizes the character in handwritten character images."""

    def __init__(self, network_fn: Type[nn.Module], network_args: Dict) -> None:
        """Intializes the CharacterModel and load the pretrained weights."""
        self.model = CharacterModel(network_fn=network_fn, network_args=network_args)
        self.model.load_weights()
        self.model.eval()

    def predict(self, image_or_filename: Union[np.ndarray, str]) -> Tuple[str, float]:
        """Predict on a single images contianing a handwritten character."""
        if isinstance(image_or_filename, str):
            image = read_image(image_or_filename, grayscale=True)
        else:
            image = image_or_filename
        return self.model.predict_on_image(image)

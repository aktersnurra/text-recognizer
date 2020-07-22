"""Defines the CharacterModel class."""
from typing import Callable, Dict, Optional, Tuple, Type

import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor

from text_recognizer.datasets.emnist_dataset import load_emnist_mapping
from text_recognizer.models.base import Model


class CharacterModel(Model):
    """Model for predicting characters from images."""

    def __init__(
        self,
        network_fn: Type[nn.Module],
        network_args: Dict,
        data_loader: Optional[Callable] = None,
        data_loader_args: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        criterion: Optional[Callable] = None,
        criterion_args: Optional[Dict] = None,
        optimizer: Optional[Callable] = None,
        optimizer_args: Optional[Dict] = None,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_args: Optional[Dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes the CharacterModel."""

        super().__init__(
            network_fn,
            network_args,
            data_loader,
            data_loader_args,
            metrics,
            criterion,
            criterion_args,
            optimizer,
            optimizer_args,
            lr_scheduler,
            lr_scheduler_args,
            device,
        )
        self.load_mapping()
        self.tensor_transform = ToTensor()
        self.softmax = nn.Softmax(dim=0)

    def load_mapping(self) -> None:
        """Mapping between integers and classes."""
        self._mapping = load_emnist_mapping()

    def predict_on_image(self, image: np.ndarray) -> Tuple[str, float]:
        """Character prediction on an image.

        Args:
            image (np.ndarray): An image containing a character.

        Returns:
            Tuple[str, float]: The predicted character and the confidence in the prediction.

        """

        if image.dtype == np.uint8:
            image = (image / 255).astype(np.float32)

        # Conver to Pytorch Tensor.
        image = self.tensor_transform(image)

        with torch.no_grad():
            logits = self.network(image)

        prediction = self.softmax(logits.data.squeeze())

        index = int(torch.argmax(prediction, dim=0))
        confidence_of_prediction = prediction[index]
        predicted_character = self._mapping[index]

        return predicted_character, confidence_of_prediction

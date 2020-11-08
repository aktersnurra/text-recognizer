"""Defines the CharacterModel class."""
from typing import Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from text_recognizer.datasets import EmnistMapper
from text_recognizer.models.base import Model


class CharacterModel(Model):
    """Model for predicting characters from images."""

    def __init__(
        self,
        network_fn: Type[nn.Module],
        dataset: Type[Dataset],
        network_args: Optional[Dict] = None,
        dataset_args: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        criterion: Optional[Callable] = None,
        criterion_args: Optional[Dict] = None,
        optimizer: Optional[Callable] = None,
        optimizer_args: Optional[Dict] = None,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_args: Optional[Dict] = None,
        swa_args: Optional[Dict] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes the CharacterModel."""

        super().__init__(
            network_fn,
            dataset,
            network_args,
            dataset_args,
            metrics,
            criterion,
            criterion_args,
            optimizer,
            optimizer_args,
            lr_scheduler,
            lr_scheduler_args,
            swa_args,
            device,
        )
        self.pad_token = dataset_args["args"]["pad_token"]
        if self._mapper is None:
            self._mapper = EmnistMapper(pad_token=self.pad_token,)
        self.tensor_transform = ToTensor()
        self.softmax = nn.Softmax(dim=0)

    @torch.no_grad()
    def predict_on_image(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[str, float]:
        """Character prediction on an image.

        Args:
            image (Union[np.ndarray, torch.Tensor]): An image containing a character.

        Returns:
            Tuple[str, float]: The predicted character and the confidence in the prediction.

        """
        self.eval()

        if image.dtype == np.uint8:
            # Converts an image with range [0, 255] with to Pytorch Tensor with range [0, 1].
            image = self.tensor_transform(image)
        if image.dtype == torch.uint8:
            # If the image is an unscaled tensor.
            image = image.type("torch.FloatTensor") / 255

        # Put the image tensor on the device the model weights are on.
        image = image.to(self.device)
        logits = self.forward(image)

        prediction = self.softmax(logits.squeeze(0))

        index = int(torch.argmax(prediction, dim=0))
        confidence_of_prediction = prediction[index]
        predicted_character = self.mapper(index)

        return predicted_character, confidence_of_prediction

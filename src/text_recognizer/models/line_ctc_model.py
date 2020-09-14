"""Defines the LineCTCModel class."""
from typing import Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from text_recognizer.datasets import EmnistMapper
from text_recognizer.models.base import Model
from text_recognizer.networks import greedy_decoder


class LineCTCModel(Model):
    """Model for predicting a sequence of characters from an image of a text line."""

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
        if self._mapper is None:
            self._mapper = EmnistMapper()
        self.tensor_transform = ToTensor()

    def loss_fn(self, output: Tensor, targets: Tensor) -> Tensor:
        """Computes the CTC loss.

        Args:
            output (Tensor): Model predictions.
            targets (Tensor): Correct output sequence.

        Returns:
            Tensor: The CTC loss.

        """

        # Input lengths on the form [T, B]
        input_lengths = torch.full(
            size=(output.shape[1],), fill_value=output.shape[0], dtype=torch.long,
        )

        # Configure target tensors for ctc loss.
        targets_ = Tensor([]).to(self.device)
        target_lengths = []
        for t in targets:
            # Remove padding symbol as it acts as the blank symbol.
            t = t[t < 79]
            targets_ = torch.cat([targets_, t])
            target_lengths.append(len(t))

        targets = targets_.type(dtype=torch.long)
        target_lengths = (
            torch.Tensor(target_lengths).type(dtype=torch.long).to(self.device)
        )

        return self.criterion(output, targets, input_lengths, target_lengths)

    @torch.no_grad()
    def predict_on_image(self, image: Union[np.ndarray, Tensor]) -> Tuple[str, float]:
        """Predict on a single input."""
        if image.dtype == np.uint8:
            # Converts an image with range [0, 255] with to Pytorch Tensor with range [0, 1].
            image = self.tensor_transform(image)

        # Rescale image between 0 and 1.
        if image.dtype == torch.uint8:
            # If the image is an unscaled tensor.
            image = image.type("torch.FloatTensor") / 255

        # Put the image tensor on the device the model weights are on.
        image = image.to(self.device)
        log_probs = (
            self.swa_network(image)
            if self.swa_network is not None
            else self.network(image)
        )

        raw_pred, _ = greedy_decoder(
            predictions=log_probs,
            character_mapper=self.mapper,
            blank_label=80,
            collapse_repeated=True,
        )

        log_probs, _ = log_probs.max(dim=2)

        predicted_characters = "".join(raw_pred[0])
        confidence_of_prediction = torch.exp(log_probs.sum()).item()

        return predicted_characters, confidence_of_prediction

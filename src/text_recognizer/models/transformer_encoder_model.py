"""Defines the CNN-Transformer class."""
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from text_recognizer.datasets import EmnistMapper
from text_recognizer.models.base import Model


class TransformerEncoderModel(Model):
    """A class for only using the encoder part in the sequence modelling."""

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
        # self.init_token = dataset_args["args"]["init_token"]
        self.pad_token = dataset_args["args"]["pad_token"]
        self.eos_token = dataset_args["args"]["eos_token"]
        if network_args is not None:
            self.max_len = network_args["max_len"]
        else:
            self.max_len = 128

        if self._mapper is None:
            self._mapper = EmnistMapper(
                # init_token=self.init_token,
                pad_token=self.pad_token,
                eos_token=self.eos_token,
            )
        self.tensor_transform = ToTensor()

        self.softmax = nn.Softmax(dim=2)

    @torch.no_grad()
    def _generate_sentence(self, image: Tensor) -> Tuple[List, float]:
        logits = self.network(image)
        # Convert logits to probabilities.
        probs = self.softmax(logits).squeeze(0)

        confidence, pred_tokens = probs.max(1)
        pred_tokens = pred_tokens

        eos_index = torch.nonzero(
            pred_tokens == self._mapper(self.eos_token), as_tuple=False,
        )

        eos_index = eos_index[0].item() if eos_index.nelement() else -1

        predicted_characters = "".join(
            [self.mapper(x) for x in pred_tokens[:eos_index].tolist()]
        )

        confidence = np.min(confidence.tolist())

        return predicted_characters, confidence

    @torch.no_grad()
    def predict_on_image(self, image: Union[np.ndarray, Tensor]) -> Tuple[str, float]:
        """Predict on a single input."""
        self.eval()

        if image.dtype == np.uint8:
            # Converts an image with range [0, 255] with to Pytorch Tensor with range [0, 1].
            image = self.tensor_transform(image)

        # Rescale image between 0 and 1.
        if image.dtype == torch.uint8:
            # If the image is an unscaled tensor.
            image = image.type("torch.FloatTensor") / 255

        # Put the image tensor on the device the model weights are on.
        image = image.to(self.device)

        (predicted_characters, confidence_of_prediction,) = self._generate_sentence(
            image
        )

        return predicted_characters, confidence_of_prediction

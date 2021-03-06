"""Defines the CNN-Transformer class."""
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset

from text_recognizer.datasets import EmnistMapper
import text_recognizer.datasets.transforms as transforms
from text_recognizer.models.base import Model
from text_recognizer.networks import greedy_decoder


class TransformerModel(Model):
    """Model for predicting a sequence of characters from an image of a text line with a cnn-transformer."""

    def __init__(
        self,
        network_fn: str,
        dataset: str,
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
        self.init_token = dataset_args["args"]["init_token"]
        self.pad_token = dataset_args["args"]["pad_token"]
        self.eos_token = dataset_args["args"]["eos_token"]
        self.lower = dataset_args["args"]["lower"]
        self.max_len = 100

        if self._mapper is None:
            self._mapper = EmnistMapper(
                init_token=self.init_token,
                pad_token=self.pad_token,
                eos_token=self.eos_token,
                lower=self.lower,
            )
        self.tensor_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.912], std=[0.168])]
        )
        self.softmax = nn.Softmax(dim=2)

    @torch.no_grad()
    def _generate_sentence(self, image: Tensor) -> Tuple[List, float]:
        src = self.network.extract_image_features(image)

        # Added for vqvae transformer.
        if isinstance(src, Tuple):
            src = src[0]

        memory = self.network.encoder(src)

        confidence_of_predictions = []
        trg_indices = [self.mapper(self.init_token)]

        for _ in range(self.max_len - 1):
            trg = torch.tensor(trg_indices, device=self.device)[None, :].long()
            trg = self.network.target_embedding(trg)
            logits = self.network.decoder(trg=trg, memory=memory, trg_mask=None)

            # Convert logits to probabilities.
            probs = self.softmax(logits)

            pred_token = probs.argmax(2)[:, -1].item()
            confidence = probs.max(2).values[:, -1].item()

            trg_indices.append(pred_token)
            confidence_of_predictions.append(confidence)

            if pred_token == self.mapper(self.eos_token):
                break

        confidence = np.min(confidence_of_predictions)
        predicted_characters = "".join([self.mapper(x) for x in trg_indices[1:]])

        return predicted_characters, confidence

    @torch.no_grad()
    def predict_on_image(self, image: Union[np.ndarray, Tensor]) -> Tuple[str, float]:
        """Predict on a single input."""
        self.eval()

        if image.dtype == np.uint8:
            # Converts an image with range [0, 255] with to PyTorch Tensor with range [0, 1].
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

"""Segmentation model for detecting and segmenting lines."""
from typing import Callable, Dict, Optional, Type, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from text_recognizer.models.base import Model


class SegmentationModel(Model):
    """Model for segmenting lines in an image."""

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
        self.tensor_transform = ToTensor()
        self.softmax = nn.Softmax(dim=2)

    @torch.no_grad()
    def predict_on_image(self, image: Union[np.ndarray, Tensor]) -> Tensor:
        """Predict on a single input."""
        self.eval()

        if image.dtype is np.uint8:
            # Converts an image with range [0, 255] with to PyTorch Tensor with range [0, 1].
            image = self.tensor_transform(image)

        # Rescale image between 0 and 1.
        if image.dtype is torch.uint8 or image.dtype is torch.int64:
            # If the image is an unscaled tensor.
            image = image.type("torch.FloatTensor") / 255

        if not torch.is_tensor(image):
            image = Tensor(image)

        # Put the image tensor on the device the model weights are on.
        image = image.to(self.device)

        logits = self.forward(image)

        segmentation_mask = torch.argmax(logits, dim=1)

        return segmentation_mask

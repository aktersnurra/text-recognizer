"""Callbacks using wandb."""
from typing import Callable, Dict, List, Optional, Type

import numpy as np
from torchvision.transforms import Compose, ToTensor
from training.callbacks import Callback
import wandb

from text_recognizer.datasets import Transpose
from text_recognizer.models.base import Model


class WandbCallback(Callback):
    """A custom W&B metric logger for the trainer."""

    def __init__(self, log_batch_frequency: int = None) -> None:
        """Short summary.

        Args:
            log_batch_frequency (int): If None, metrics will be logged every epoch.
                If set to an integer, callback will log every metrics every log_batch_frequency.

        """
        super().__init__()
        self.log_batch_frequency = log_batch_frequency

    def _on_batch_end(self, batch: int, logs: Dict) -> None:
        if self.log_batch_frequency and batch % self.log_batch_frequency == 0:
            wandb.log(logs, commit=True)

    def on_train_batch_end(self, batch: int, logs: Dict = {}) -> None:
        """Logs training metrics."""
        if logs is not None:
            self._on_batch_end(batch, logs)

    def on_validation_batch_end(self, batch: int, logs: Dict = {}) -> None:
        """Logs validation metrics."""
        if logs is not None:
            self._on_batch_end(batch, logs)

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        """Logs at epoch end."""
        wandb.log(logs, commit=True)


class WandbImageLogger(Callback):
    """Custom W&B callback for image logging."""

    def __init__(
        self,
        example_indices: Optional[List] = None,
        num_examples: int = 4,
        transfroms: Optional[Callable] = None,
    ) -> None:
        """Initializes the WandbImageLogger with the model to train.

        Args:
            example_indices (Optional[List]): Indices for validation images. Defaults to None.
            num_examples (int): Number of random samples to take if example_indices are not specified. Defaults to 4.
            transfroms (Optional[Callable]): Transforms to use on the validation images, e.g. transpose. Defaults to
                None.

        """

        super().__init__()
        self.example_indices = example_indices
        self.num_examples = num_examples
        self.transfroms = transfroms
        if self.transfroms is None:
            self.transforms = Compose([Transpose()])

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and extracts validation images from the dataset."""
        self.model = model
        data_loader = self.model.data_loaders["val"]
        if self.example_indices is None:
            self.example_indices = np.random.randint(
                0, len(data_loader.dataset.data), self.num_examples
            )
        self.val_images = data_loader.dataset.data[self.example_indices]
        self.val_targets = data_loader.dataset.targets[self.example_indices].numpy()

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        """Get network predictions on validation images."""
        images = []
        for i, image in enumerate(self.val_images):
            image = self.transforms(image)
            pred, conf = self.model.predict_on_image(image)
            ground_truth = self.model.mapper(int(self.val_targets[i]))
            caption = f"Prediction: {pred} Confidence: {conf:.3f} Ground Truth: {ground_truth}"
            images.append(wandb.Image(image, caption=caption))

        wandb.log({"examples": images}, commit=False)

"""Callback for W&B."""
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import torch
from torchvision.transforms import ToTensor
from training.trainer.callbacks import Callback
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

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Logs training metrics."""
        if logs is not None:
            logs["lr"] = self.model.optimizer.param_groups[0]["lr"]
            self._on_batch_end(batch, logs)

    def on_validation_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
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
        use_transpose: Optional[bool] = False,
    ) -> None:
        """Initializes the WandbImageLogger with the model to train.

        Args:
            example_indices (Optional[List]): Indices for validation images. Defaults to None.
            num_examples (int): Number of random samples to take if example_indices are not specified. Defaults to 4.
            use_transpose (Optional[bool]): Use transpose on image or not. Defaults to False.

        """

        super().__init__()
        self.caption = None
        self.example_indices = example_indices
        self.test_sample_indices = None
        self.num_examples = num_examples
        self.transpose = Transpose() if use_transpose else None

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and extracts validation images from the dataset."""
        self.model = model
        self.caption = "Validation Examples"
        if self.example_indices is None:
            self.example_indices = np.random.randint(
                0, len(self.model.val_dataset), self.num_examples
            )
        self.images = self.model.val_dataset.dataset.data[self.example_indices]
        self.targets = self.model.val_dataset.dataset.targets[self.example_indices]
        self.targets = self.targets.tolist()

    def on_test_begin(self) -> None:
        """Get samples from test dataset."""
        self.caption = "Test Examples"
        if self.test_sample_indices is None:
            self.test_sample_indices = np.random.randint(
                0, len(self.model.test_dataset), self.num_examples
            )
        self.images = self.model.test_dataset.data[self.test_sample_indices]
        self.targets = self.model.test_dataset.targets[self.test_sample_indices]
        self.targets = self.targets.tolist()

    def on_test_end(self) -> None:
        """Log test images."""
        self.on_epoch_end(0, {})

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        """Get network predictions on validation images."""
        images = []
        for i, image in enumerate(self.images):
            image = self.transpose(image) if self.transpose is not None else image
            pred, conf = self.model.predict_on_image(image)
            if isinstance(self.targets[i], list):
                ground_truth = "".join(
                    [
                        self.model.mapper(int(target_index))
                        for target_index in self.targets[i]
                    ]
                ).rstrip("_")
            else:
                ground_truth = self.targets[i]
            caption = f"Prediction: {pred} Confidence: {conf:.3f} Ground Truth: {ground_truth}"
            images.append(wandb.Image(image, caption=caption))

        wandb.log({f"{self.caption}": images}, commit=False)

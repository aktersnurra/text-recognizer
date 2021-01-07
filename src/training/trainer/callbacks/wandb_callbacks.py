"""Callback for W&B."""
from typing import Callable, Dict, List, Optional, Type

import numpy as np
from training.trainer.callbacks import Callback
import wandb

import text_recognizer.datasets.transforms as transforms
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
        transform: Optional[bool] = None,
    ) -> None:
        """Initializes the WandbImageLogger with the model to train.

        Args:
            example_indices (Optional[List]): Indices for validation images. Defaults to None.
            num_examples (int): Number of random samples to take if example_indices are not specified. Defaults to 4.
            transform (Optional[Dict]): Use transform on image or not. Defaults to None.

        """

        super().__init__()
        self.caption = None
        self.example_indices = example_indices
        self.test_sample_indices = None
        self.num_examples = num_examples
        self.transform = (
            self._configure_transform(transform) if transform is not None else None
        )

    def _configure_transform(self, transform: Dict) -> Callable:
        args = transform["args"] or {}
        return getattr(transforms, transform["type"])(**args)

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
            image = self.transform(image) if self.transform is not None else image
            pred, conf = self.model.predict_on_image(image)
            if isinstance(self.targets[i], list):
                ground_truth = "".join(
                    [
                        self.model.mapper(int(target_index) - 26)
                        if target_index > 35
                        else self.model.mapper(int(target_index))
                        for target_index in self.targets[i]
                    ]
                ).rstrip("_")
            else:
                ground_truth = self.model.mapper(int(self.targets[i]))
            caption = f"Prediction: {pred} Confidence: {conf:.3f} Ground Truth: {ground_truth}"
            images.append(wandb.Image(image, caption=caption))

        wandb.log({f"{self.caption}": images}, commit=False)


class WandbSegmentationLogger(Callback):
    """Custom W&B callback for image logging."""

    def __init__(
        self,
        class_labels: Dict,
        example_indices: Optional[List] = None,
        num_examples: int = 4,
    ) -> None:
        """Initializes the WandbImageLogger with the model to train.

        Args:
            class_labels (Dict): A dict with int as key and class string as value.
            example_indices (Optional[List]): Indices for validation images. Defaults to None.
            num_examples (int): Number of random samples to take if example_indices are not specified. Defaults to 4.

        """

        super().__init__()
        self.caption = None
        self.class_labels = {int(k): v for k, v in class_labels.items()}
        self.example_indices = example_indices
        self.test_sample_indices = None
        self.num_examples = num_examples

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and extracts validation images from the dataset."""
        self.model = model
        self.caption = "Validation Segmentation Examples"
        if self.example_indices is None:
            self.example_indices = np.random.randint(
                0, len(self.model.val_dataset), self.num_examples
            )
        self.images = self.model.val_dataset.dataset.data[self.example_indices]
        self.targets = self.model.val_dataset.dataset.targets[self.example_indices]
        self.targets = self.targets.tolist()

    def on_test_begin(self) -> None:
        """Get samples from test dataset."""
        self.caption = "Test Segmentation Examples"
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
            pred_mask = (
                self.model.predict_on_image(image).detach().squeeze(0).cpu().numpy()
            )
            gt_mask = np.array(self.targets[i])
            images.append(
                wandb.Image(
                    image,
                    masks={
                        "predictions": {
                            "mask_data": pred_mask,
                            "class_labels": self.class_labels,
                        },
                        "ground_truth": {
                            "mask_data": gt_mask,
                            "class_labels": self.class_labels,
                        },
                    },
                )
            )

        wandb.log({f"{self.caption}": images}, commit=False)

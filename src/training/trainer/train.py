"""Training script for PyTorch models."""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Type
import warnings

from einops import rearrange
from loguru import logger
import numpy as np
import torch
from torch import Tensor
from torch.optim.swa_utils import update_bn
from training.trainer.callbacks import Callback, CallbackList, LRScheduler, SWA
from training.trainer.util import log_val_metric
import wandb

from text_recognizer.models import Model


torch.backends.cudnn.benchmark = True
np.random.seed(4711)
torch.manual_seed(4711)
torch.cuda.manual_seed(4711)


warnings.filterwarnings("ignore")


class Trainer:
    """Trainer for training PyTorch models."""

    def __init__(
        self,
        max_epochs: int,
        callbacks: List[Type[Callback]],
        transformer_model: bool = False,
        max_norm: float = 0.0,
        freeze_backbone: Optional[int] = None,
    ) -> None:
        """Initialization of the Trainer.

        Args:
            max_epochs (int): The maximum number of epochs in the training loop.
            callbacks (CallbackList): List of callbacks to be called.
            transformer_model (bool): Transformer model flag, modifies the input to the model. Default is False.
            max_norm (float): Max norm for gradient cl:ipping. Defaults to 0.0.
            freeze_backbone (Optional[int]): How many epochs to freeze the backbone for. Used when training
                Transformers. Default is None.

        """
        # Training arguments.
        self.start_epoch = 1
        self.max_epochs = max_epochs
        self.callbacks = callbacks
        self.freeze_backbone = freeze_backbone

        # Flag for setting callbacks.
        self.callbacks_configured = False

        self.transformer_model = transformer_model

        self.max_norm = max_norm

        # Model placeholders
        self.model = None

    def _configure_callbacks(self) -> None:
        """Instantiate the CallbackList."""
        if not self.callbacks_configured:
            # If learning rate schedulers are present, they need to be added to the callbacks.
            if self.model.swa_scheduler is not None:
                self.callbacks.append(SWA())
            elif self.model.lr_scheduler is not None:
                self.callbacks.append(LRScheduler())

            self.callbacks = CallbackList(self.model, self.callbacks)

    def compute_metrics(
        self, output: Tensor, targets: Tensor, loss: Tensor, batch_size: int
    ) -> Dict:
        """Computes metrics for output and target pairs."""
        # Compute metrics.
        loss = loss.detach().float().item()
        output = output.detach()
        targets = targets.detach()
        if self.model.metrics is not None:
            metrics = {}
            for metric in self.model.metrics:
                if metric == "cer" or metric == "wer":
                    metrics[metric] = self.model.metrics[metric](
                        output,
                        targets,
                        batch_size,
                        self.model.mapper(self.model.pad_token),
                    )
                else:
                    metrics[metric] = self.model.metrics[metric](output, targets)
        else:
            metrics = {}
        metrics["loss"] = loss

        return metrics

    def training_step(self, batch: int, samples: Tuple[Tensor, Tensor],) -> Dict:
        """Performs the training step."""
        # Pass the tensor to the device for computation.
        data, targets = samples
        data, targets = (
            data.to(self.model.device),
            targets.to(self.model.device),
        )

        batch_size = data.shape[0]

        # Placeholder for uxiliary loss.
        aux_loss = None

        # Forward pass.
        # Get the network prediction.
        if self.transformer_model:
            if self.freeze_backbone is not None and batch < self.freeze_backbone:
                with torch.no_grad():
                    image_features = self.model.network.extract_image_features(data)

                if isinstance(image_features, Tuple):
                    image_features, _ = image_features

                output = self.model.network.decode_image_features(
                    image_features, targets[:, :-1]
                )
            else:
                output = self.model.network.forward(data, targets[:, :-1])
                if isinstance(output, Tuple):
                    output, aux_loss = output
            output = rearrange(output, "b t v -> (b t) v")
            targets = rearrange(targets[:, 1:], "b t -> (b t)").long()
        else:
            output = self.model.forward(data)

            if isinstance(output, Tuple):
                output, aux_loss = output
                targets = data

        # Compute the loss.
        loss = self.model.criterion(output, targets)

        if aux_loss is not None:
            loss += aux_loss

        # Backward pass.
        # Clear the previous gradients.
        for p in self.model.network.parameters():
            p.grad = None

        # Compute the gradients.
        loss.backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.network.parameters(), self.max_norm
            )

        # Perform updates using calculated gradients.
        self.model.optimizer.step()

        metrics = self.compute_metrics(output, targets, loss, batch_size)

        return metrics

    def train(self) -> None:
        """Runs the training loop for one epoch."""
        # Set model to traning mode.
        self.model.train()

        for batch, samples in enumerate(self.model.train_dataloader()):
            self.callbacks.on_train_batch_begin(batch)
            metrics = self.training_step(batch, samples)
            self.callbacks.on_train_batch_end(batch, logs=metrics)

    @torch.no_grad()
    def validation_step(self, batch: int, samples: Tuple[Tensor, Tensor],) -> Dict:
        """Performs the validation step."""

        # Pass the tensor to the device for computation.
        data, targets = samples
        data, targets = (
            data.to(self.model.device),
            targets.to(self.model.device),
        )

        batch_size = data.shape[0]

        # Placeholder for uxiliary loss.
        aux_loss = None

        # Forward pass.
        # Get the network prediction.
        # Use SWA if available and using test dataset.
        if self.transformer_model:
            output = self.model.network.forward(data, targets[:, :-1])
            if isinstance(output, Tuple):
                output, aux_loss = output
            output = rearrange(output, "b t v -> (b t) v")
            targets = rearrange(targets[:, 1:], "b t -> (b t)").long()
        else:
            output = self.model.forward(data)

            if isinstance(output, Tuple):
                output, aux_loss = output
                targets = data

        # Compute the loss.
        loss = self.model.criterion(output, targets)

        if aux_loss is not None:
            loss += aux_loss

        # Compute metrics.
        metrics = self.compute_metrics(output, targets, loss, batch_size)

        return metrics

    def validate(self) -> Dict:
        """Runs the validation loop for one epoch."""
        # Set model to eval mode.
        self.model.eval()

        # Summary for the current eval loop.
        summary = []

        for batch, samples in enumerate(self.model.val_dataloader()):
            self.callbacks.on_validation_batch_begin(batch)
            metrics = self.validation_step(batch, samples)
            self.callbacks.on_validation_batch_end(batch, logs=metrics)
            summary.append(metrics)

        # Compute mean of all metrics.
        metrics_mean = {
            "val_" + metric: np.mean([x[metric] for x in summary])
            for metric in summary[0]
        }

        return metrics_mean

    def fit(self, model: Type[Model]) -> None:
        """Runs the training and evaluation loop."""

        # Sets model, loads the data, criterion, and optimizers.
        self.model = model
        self.model.prepare_data()
        self.model.configure_model()

        # Configure callbacks.
        self._configure_callbacks()

        # Set start time.
        t_start = time.time()

        self.callbacks.on_fit_begin()

        # Run the training loop.
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.callbacks.on_epoch_begin(epoch)

            # Perform one training pass over the training set.
            self.train()

            # Evaluate the model on the validation set.
            val_metrics = self.validate()
            log_val_metric(val_metrics, epoch)

            self.callbacks.on_epoch_end(epoch, logs=val_metrics)

            if self.model.stop_training:
                break

        # Calculate the total training time.
        t_end = time.time()
        t_training = t_end - t_start

        self.callbacks.on_fit_end()

        logger.info(f"Training took {t_training:.2f} s.")

        # "Teardown".
        self.model = None

    def test(self, model: Type[Model]) -> Dict:
        """Run inference on test data."""

        # Sets model, loads the data, criterion, and optimizers.
        self.model = model
        self.model.prepare_data()
        self.model.configure_model()

        # Configure callbacks.
        self._configure_callbacks()

        self.callbacks.on_test_begin()

        self.model.eval()

        # Check if SWA network is available.
        self.model.use_swa_model()

        # Summary for the current test loop.
        summary = []

        for batch, samples in enumerate(self.model.test_dataloader()):
            metrics = self.validation_step(batch, samples)
            summary.append(metrics)

        self.callbacks.on_test_end()

        # Compute mean of all test metrics.
        metrics_mean = {
            "test_" + metric: np.mean([x[metric] for x in summary])
            for metric in summary[0]
        }

        # "Teardown".
        self.model = None

        return metrics_mean

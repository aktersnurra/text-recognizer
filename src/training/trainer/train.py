"""Training script for PyTorch models."""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Type

from loguru import logger
import numpy as np
import torch
from torch import Tensor
from training.trainer.callbacks import Callback, CallbackList
from training.trainer.util import RunningAverage
import wandb

from text_recognizer.models import Model


torch.backends.cudnn.benchmark = True
np.random.seed(4711)
torch.manual_seed(4711)
torch.cuda.manual_seed(4711)


class Trainer:
    """Trainer for training PyTorch models."""

    def __init__(
        self,
        model: Type[Model],
        model_dir: Path,
        train_args: Dict,
        callbacks: CallbackList,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        """Initialization of the Trainer.

        Args:
            model (Type[Model]): A model object.
            model_dir (Path): Path to the model directory.
            train_args (Dict): The training arguments.
            callbacks (CallbackList): List of callbacks to be called.
            checkpoint_path (Optional[Path]): The path to a previously trained model. Defaults to None.

        """
        self.model = model
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
        self.start_epoch = 1
        self.epochs = train_args["epochs"]
        self.callbacks = callbacks

        if self.checkpoint_path is not None:
            self.start_epoch = self.model.load_checkpoint(self.checkpoint_path)

        # Parse the name of the experiment.
        experiment_dir = str(self.model_dir.parents[1]).split("/")
        self.experiment_name = experiment_dir[-2] + "/" + experiment_dir[-1]

    def training_step(
        self,
        batch: int,
        samples: Tuple[Tensor, Tensor],
        loss_avg: Type[RunningAverage],
    ) -> Dict:
        """Performs the training step."""
        # Pass the tensor to the device for computation.
        data, targets = samples
        data, targets = (
            data.to(self.model.device),
            targets.to(self.model.device),
        )

        # Forward pass.
        # Get the network prediction.
        output = self.model.network(data)

        # Compute the loss.
        loss = self.model.criterion(output, targets)

        # Backward pass.
        # Clear the previous gradients.
        self.model.optimizer.zero_grad()

        # Compute the gradients.
        loss.backward()

        # Perform updates using calculated gradients.
        self.model.optimizer.step()

        # Compute metrics.
        loss_avg.update(loss.item())
        output = output.data.cpu()
        targets = targets.data.cpu()
        metrics = {
            metric: self.model.metrics[metric](output, targets)
            for metric in self.model.metrics
        }
        metrics["loss"] = loss_avg()
        return metrics

    def train(self) -> None:
        """Runs the training loop for one epoch."""
        # Set model to traning mode.
        self.model.train()

        # Running average for the loss.
        loss_avg = RunningAverage()

        data_loader = self.model.data_loaders["train"]

        for batch, samples in enumerate(data_loader):
            self.callbacks.on_train_batch_begin(batch)
            metrics = self.training_step(batch, samples, loss_avg)
            self.callbacks.on_train_batch_end(batch, logs=metrics)

    @torch.no_grad()
    def validation_step(
        self,
        batch: int,
        samples: Tuple[Tensor, Tensor],
        loss_avg: Type[RunningAverage],
    ) -> Dict:
        """Performs the validation step."""
        # Pass the tensor to the device for computation.
        data, targets = samples
        data, targets = (
            data.to(self.model.device),
            targets.to(self.model.device),
        )

        # Forward pass.
        # Get the network prediction.
        output = self.model.network(data)

        # Compute the loss.
        loss = self.model.criterion(output, targets)

        # Compute metrics.
        loss_avg.update(loss.item())
        output = output.data.cpu()
        targets = targets.data.cpu()
        metrics = {
            metric: self.model.metrics[metric](output, targets)
            for metric in self.model.metrics
        }
        metrics["loss"] = loss.item()

        return metrics

    def _log_val_metric(self, metrics_mean: Dict, epoch: Optional[int] = None) -> None:
        log_str = "Validation metrics " + (f"at epoch {epoch} - " if epoch else " - ")
        logger.debug(
            log_str + " - ".join(f"{k}: {v:.4f}" for k, v in metrics_mean.items())
        )

    def validate(self, epoch: Optional[int] = None) -> Dict:
        """Runs the validation loop for one epoch."""
        # Set model to eval mode.
        self.model.eval()

        # Running average for the loss.
        data_loader = self.model.data_loaders["val"]

        # Running average for the loss.
        loss_avg = RunningAverage()

        # Summary for the current eval loop.
        summary = []

        for batch, samples in enumerate(data_loader):
            self.callbacks.on_validation_batch_begin(batch)
            metrics = self.validation_step(batch, samples, loss_avg)
            self.callbacks.on_validation_batch_end(batch, logs=metrics)
            summary.append(metrics)

        # Compute mean of all metrics.
        metrics_mean = {
            "val_" + metric: np.mean([x[metric] for x in summary])
            for metric in summary[0]
        }
        self._log_val_metric(metrics_mean, epoch)

        return metrics_mean

    def fit(self) -> None:
        """Runs the training and evaluation loop."""

        logger.debug(f"Running an experiment called {self.experiment_name}.")

        # Set start time.
        t_start = time.time()

        self.callbacks.on_fit_begin()

        # Run the training loop.
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.callbacks.on_epoch_begin(epoch)

            # Perform one training pass over the training set.
            self.train()

            # Evaluate the model on the validation set.
            val_metrics = self.validate(epoch)

            self.callbacks.on_epoch_end(epoch, logs=val_metrics)

            if self.model.stop_training:
                break

        # Calculate the total training time.
        t_end = time.time()
        t_training = t_end - t_start

        self.callbacks.on_fit_end()

        logger.info(f"Training took {t_training:.2f} s.")

"""Training script for PyTorch models."""

from pathlib import Path
import time
from typing import Dict, Optional, Type

from loguru import logger
import numpy as np
import torch
from tqdm import tqdm, trange
from training.util import RunningAverage
import wandb

from text_recognizer.models import Model

torch.backends.cudnn.benchmark = True
np.random.seed(4711)
torch.manual_seed(4711)
torch.cuda.manual_seed(4711)


class Trainer:
    """Trainer for training PyTorch models."""

    # TODO implement wandb.
    # TODO implement Bayesian parameter search.

    def __init__(
        self,
        model: Type[Model],
        model_dir: Path,
        epochs: int,
        val_metric: str = "accuracy",
        checkpoint_path: Optional[Path] = None,
        use_wandb: Optional[bool] = False,
    ) -> None:
        """Initialization of the Trainer.

        Args:
            model (Type[Model]): A model object.
            model_dir (Path): Path to the model directory.
            epochs (int): Number of epochs to train.
            val_metric (str): The validation metric to evaluate the model on. Defaults to "accuracy".
            checkpoint_path (Optional[Path]): The path to a previously trained model. Defaults to None.
            use_wandb (Optional[bool]): Sync training to wandb.

        """
        self.model = model
        self.model_dir = model_dir
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.start_epoch = 0

        if self.checkpoint_path is not None:
            self.start_epoch = self.model.load_checkpoint(self.checkpoint_path)

        if use_wandb:
            # TODO implement wandb logging.
            pass

        self.val_metric = val_metric
        self.best_val_metric = 0.0

        # Parse the name of the experiment.
        experiment_dir = str(self.model_dir.parents[1]).split("/")
        self.experiment_name = experiment_dir[-2] + "/" + experiment_dir[-1]

    def train(self) -> None:
        """Training loop."""
        # Set model to traning mode.
        self.model.train()

        # Running average for the loss.
        loss_avg = RunningAverage()

        data_loader = self.model.data_loaders("train")

        with tqdm(
            total=len(data_loader),
            leave=False,
            unit="step",
            bar_format="{n_fmt}/{total_fmt} |{bar:20}| {remaining} {rate_inv_fmt}{postfix}",
        ) as t:
            for data, targets in data_loader:

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

                # Update Tqdm progress bar.
                t.set_postfix(**metrics)
                t.update()

                # If the model has a learning rate scheduler, compute a step.
                if self.model.lr_scheduler is not None:
                    self.model.lr_scheduler.step()

    def validate(self) -> Dict:
        """Evaluation loop.

        Returns:
            Dict: A dictionary of evaluation metrics.

        """
        # Set model to eval mode.
        self.model.eval()

        # Running average for the loss.
        data_loader = self.model.data_loaders("val")

        # Running average for the loss.
        loss_avg = RunningAverage()

        # Summary for the current eval loop.
        summary = []

        with tqdm(
            total=len(data_loader),
            leave=False,
            unit="step",
            bar_format="{n_fmt}/{total_fmt} |{bar:20}| {remaining} {rate_inv_fmt}{postfix}",
        ) as t:
            for data, targets in data_loader:
                data, targets = (
                    data.to(self.model.device),
                    targets.to(self.model.device),
                )

                with torch.no_grad():
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

                summary.append(metrics)

                # Update Tqdm progress bar.
                t.set_postfix(**metrics)
                t.update()

        # Compute mean of all metrics.
        metrics_mean = {
            metric: np.mean([x[metric] for x in summary]) for metric in summary[0]
        }
        metrics_str = " - ".join(f"{k}: {v}" for k, v in metrics_mean.items())
        logger.debug(metrics_str)

        return metrics_mean

    def fit(self) -> None:
        """Runs the training and evaluation loop."""

        logger.debug(f"Running an experiment called {self.experiment_name}.")
        t_start = time.time()
        # Run the training loop.
        for epoch in trange(
            self.epochs,
            initial=self.start_epoch,
            leave=False,
            bar_format="{desc}: {n_fmt}/{total_fmt} |{bar:10}| {remaining}{postfix}",
            desc="Epoch",
        ):
            # Perform one training pass over the training set.
            self.train()

            # Evaluate the model on the validation set.
            val_metrics = self.validate()

            # The validation metric to evaluate the model on, e.g. accuracy.
            val_metric = val_metrics[self.val_metric]
            is_best = val_metric >= self.best_val_metric
            self.best_val_metric = val_metric if is_best else self.best_val_metric
            # Save checkpoint.
            self.model.save_checkpoint(self.model_dir, is_best, epoch, self.val_metric)

            if self.start_epoch > 0 and epoch + self.start_epoch == self.epochs:
                logger.debug(f"Trained the model for {self.epochs} number of epochs.")
                break

        t_end = time.time()
        t_training = t_end - t_start
        logger.info(f"Training took {t_training:.2f} s.")

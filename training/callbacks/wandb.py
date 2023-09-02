"""Weights and Biases callbacks."""
from pathlib import Path
from typing import Tuple

import wandb
from torch import Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get W&B logger from Trainer."""

    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            return logger

    raise Exception("Weight and Biases logger not found for some reason...")


class WatchModel(Callback):
    """Make W&B watch the model at the beginning of the run."""

    def __init__(
        self,
        log_params: str = "gradients",
        log_freq: int = 100,
        log_graph: bool = False,
    ) -> None:
        self.log_params = log_params
        self.log_freq = log_freq
        self.log_graph = log_graph

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Watches model weights with wandb."""
        logger = get_wandb_logger(trainer)
        logger.watch(
            model=trainer.model,
            log=self.log_params,
            log_freq=self.log_freq,
            log_graph=self.log_graph,
        )


class UploadConfigAsArtifact(Callback):
    """Upload all *.py files to W&B as an artifact, at the beginning of the run."""

    def __init__(self) -> None:
        self.config_dir = Path(".hydra/")

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Uploads project code as an artifact."""
        logger = get_wandb_logger(trainer)
        experiment = logger.experiment
        artifact = wandb.Artifact("experiment-config", type="config")
        for filepath in self.config_dir.rglob("*.yaml"):
            artifact.add_file(str(filepath))

        experiment.use_artifact(artifact)


class ImageToCaption(Callback):
    """Logs the image and output caption."""

    def __init__(self, num_samples: int = 8, on_train: bool = True) -> None:
        self.num_samples = num_samples
        self.on_train = on_train
        self._required_keys = ("predictions", "ground_truths")

    def _log_captions(
        self, trainer: Trainer, batch: Tuple[Tensor, Tensor], outputs: dict, key: str
    ) -> None:
        xs, _ = batch
        preds, gts = outputs["predictions"], outputs["ground_truths"]
        xs, preds, gts = (
            list(xs[: self.num_samples]),
            preds[: self.num_samples],
            gts[: self.num_samples],
        )
        trainer.logger.log_image(key, xs, caption=preds)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: Tuple[Tensor, Tensor],
        *args,
    ) -> None:
        """Logs predictions on validation batch end."""
        if self.has_metrics(outputs):
            self._log_captions(trainer, batch, outputs, "train/predictions")

    @rank_zero_only
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: Tuple[Tensor, Tensor],
        *args,
    ) -> None:
        """Logs predictions on validation batch end."""
        if self.has_metrics(outputs):
            self._log_captions(trainer, batch, outputs, "val/predictions")

    @rank_zero_only
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: Tuple[Tensor, Tensor],
        *args,
    ) -> None:
        """Logs predictions on train batch end."""
        if self.has_metrics(outputs):
            self._log_captions(trainer, batch, outputs, "test/predictions")

    def has_metrics(self, outputs: dict) -> bool:
        return all(k in outputs.keys() for k in self._required_keys)

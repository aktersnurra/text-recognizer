"""Weights and Biases callbacks."""
from pathlib import Path
from typing import List

import attr
import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get W&B logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception("Weight and Biases logger not found for some reason...")


@attr.s
class WatchModel(Callback):
    """Make W&B watch the model at the beginning of the run."""

    log: str = attr.ib(default="gradients")
    log_freq: int = attr.ib(default=100)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Watches model weights with wandb."""
        logger = get_wandb_logger(trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


@attr.s
class UploadCodeAsArtifact(Callback):
    """Upload all *.py files to W&B as an artifact, at the beginning of the run."""

    project_dir: Path = attr.ib(converter=Path)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Uploads project code as an artifact."""
        logger = get_wandb_logger(trainer)
        experiment = logger.experiment
        artifact = wandb.Artifact("project-source", type="code")
        for filepath in self.project_dir.glob("**/*.py"):
            artifact.add_file(filepath)

        experiment.use_artifact(artifact)


@attr.s
class UploadCheckpointAsArtifact(Callback):
    """Upload checkpoint to wandb as an artifact, at the end of a run."""

    ckpt_dir: Path = attr.ib(converter=Path)
    upload_best_only: bool = attr.ib()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Uploads model checkpoint to W&B."""
        logger = get_wandb_logger(trainer)
        experiment = logger.experiment
        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for ckpt in (self.ckpt_dir).glob("**/*.ckpt"):
                ckpts.add_file(ckpt)

        experiment.use_artifact(ckpts)


@attr.s
class LogTextPredictions(Callback):
    """Logs a validation batch with image to text transcription."""

    num_samples: int = attr.ib(default=8)
    ready: bool = attr.ib(default=True)

    def __attrs_pre_init__(self):
        super().__init__()

    def on_sanity_check_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Sets ready attribute."""
        self.ready = False

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Logs predictions on validation epoch end."""
        if not self.ready:
            return None

        logger = get_wandb_logger(trainer)
        experiment = logger.experiment

        # Get a validation batch from the validation dataloader.
        samples = next(iter(trainer.datamodule.val_dataloader()))
        imgs, labels = samples

        imgs = imgs.to(device=pl_module.device)
        logits = pl_module(imgs)

        mapping = pl_module.mapping
        experiment.log(
            {
                f"Images/{experiment.name}": [
                    wandb.Image(
                        img,
                        caption=f"Pred: {mapping.get_text(pred)}, Label: {mapping.get_text(label)}",
                    )
                    for img, pred, label in zip(
                        imgs[: self.num_samples],
                        logits[: self.num_samples],
                        labels[: self.num_samples],
                    )
                ]
            }
        )
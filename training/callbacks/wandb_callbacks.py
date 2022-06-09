"""Weights and Biases callbacks."""
from pathlib import Path

import wandb
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get W&B logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception("Weight and Biases logger not found for some reason...")


class WatchModel(Callback):
    """Make W&B watch the model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100) -> None:
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Watches model weights with wandb."""
        logger = get_wandb_logger(trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


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


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoint to wandb as an artifact, at the end of a run."""

    def __init__(
        self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Uploads model checkpoint to W&B."""
        logger = get_wandb_logger(trainer)
        experiment = logger.experiment
        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for ckpt in (self.ckpt_dir).rglob("**/*.ckpt"):
                ckpts.add_file(ckpt)

        experiment.use_artifact(ckpts)


class LogTextPredictions(Callback):
    """Logs a validation batch with image to text transcription."""

    def __init__(self, num_samples: int = 8) -> None:
        self.num_samples = num_samples
        self.ready = False

    def _log_predictions(
        self,
        stage: str,
        trainer: Trainer,
        pl_module: LightningModule,
        dataloader: DataLoader,
    ) -> None:
        """Logs the predicted text contained in the images."""
        if not self.ready:
            return None

        logger = get_wandb_logger(trainer)
        experiment = logger.experiment

        # Get a validation batch from the validation dataloader.
        samples = next(iter(dataloader))
        imgs, labels = samples

        imgs = imgs.to(device=pl_module.device)
        logits = pl_module.predict(imgs)

        mapping = pl_module.mapping
        data = [
            wandb.Image(img, caption=mapping.get_text(pred))
            for img, pred, label in zip(
                imgs[: self.num_samples],
                logits[: self.num_samples],
                labels[: self.num_samples],
            )
        ]

        experiment.log({f"HTR/{experiment.name}/{stage}": data})

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
        dataloader = trainer.datamodule.val_dataloader()
        self._log_predictions(
            stage="val", trainer=trainer, pl_module=pl_module, dataloader=dataloader
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Logs predictions on train epoch end."""
        dataloader = trainer.datamodule.test_dataloader()
        self._log_predictions(
            stage="test", trainer=trainer, pl_module=pl_module, dataloader=dataloader
        )


class LogReconstuctedImages(Callback):
    """Log reconstructions of images."""

    def __init__(self, num_samples: int = 8, use_sigmoid: bool = False) -> None:
        self.num_samples = num_samples
        self.ready = False
        self.sigmoid = nn.Sigmoid() if use_sigmoid else None

    def _log_reconstruction(
        self,
        stage: str,
        trainer: Trainer,
        pl_module: LightningModule,
        dataloader: DataLoader,
    ) -> None:
        """Logs the reconstructions."""
        if not self.ready:
            return None

        logger = get_wandb_logger(trainer)
        experiment = logger.experiment

        # Get a validation batch from the validation dataloader.
        samples = next(iter(dataloader))
        imgs, _ = samples

        imgs = imgs.to(device=pl_module.device)
        reconstructions = pl_module(imgs)[0]
        reconstructions = (
            self.sigmoid(reconstructions)
            if self.sigmoid is not None
            else reconstructions
        )

        data = [
            wandb.Image(
                make_grid([img, rec]), caption="Left: Image, Right: Reconstruction"
            )
            # wandb.Image(rec, caption="Reconstruction"),
            for img, rec in zip(
                imgs[: self.num_samples], reconstructions[: self.num_samples]
            )
        ]

        experiment.log({f"Reconstructions/{experiment.name}/{stage}": data})

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
        dataloader = trainer.datamodule.val_dataloader()
        self._log_reconstruction(
            stage="val", trainer=trainer, pl_module=pl_module, dataloader=dataloader
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Logs predictions on train epoch end."""
        dataloader = trainer.datamodule.test_dataloader()
        self._log_reconstruction(
            stage="test", trainer=trainer, pl_module=pl_module, dataloader=dataloader
        )

"""Base PyTorch Lightning model."""
from typing import Any, Dict, Tuple, Type

import madgrad
import pytorch_lightning as pl
import torch
from torch import nn
from torch import Tensor
import torchmetrics

from text_recognizer import networks


class LitBaseModel(pl.LightningModule):
    """Abstract PyTorch Lightning class."""

    def __init__(
        self,
        network_args: Dict,
        optimizer_args: Dict,
        lr_scheduler_args: Dict,
        criterion_args: Dict,
        monitor: str = "val_loss",
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.network = getattr(networks, network_args["type"])(**network_args["args"])
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.loss_fn = self.configure_criterion(criterion_args)

        # Accuracy metric
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    @staticmethod
    def configure_criterion(criterion_args: Dict) -> Type[nn.Module]:
        """Returns a loss functions."""
        args = {} or criterion_args["args"]
        return getattr(nn, criterion_args["type"])(**args)

    def configure_optimizer(self) -> Dict[str, Any]:
        """Configures optimizer and lr scheduler."""
        args = {} or self.optimizer_args["args"]
        if self.optimizer_args["type"] == "MADGRAD":
            optimizer = getattr(madgrad, self.optimizer_args["type"])(**args)
        else:
            optimizer = getattr(torch.optim, self.optimizer_args["type"])(**args)

        args = {} or self.lr_scheduler_args["args"]
        scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_args["type"])(
            **args
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.monitor,
        }

    def forward(self, data: Tensor) -> Tensor:
        """Feedforward pass."""
        return self.network(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits = self(data)
        loss = self.loss_fn(logits, targets)
        self.log("train_loss", loss)
        self.train_acc(logits, targets)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, targets = batch
        logits = self(data)
        loss = self.loss_fn(logits, targets)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits, targets)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch
        logits = self(data)
        self.test_acc(logits, targets)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

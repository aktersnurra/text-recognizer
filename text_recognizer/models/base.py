"""Base PyTorch Lightning model."""
from typing import Any, Dict, List, Union, Tuple, Type

import madgrad
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from torch import nn
from torch import Tensor
import torchmetrics


class LitBaseModel(pl.LightningModule):
    """Abstract PyTorch Lightning class."""

    def __init__(
        self,
        network: Type[nn.Module],
        optimizer: Union[DictConfig, Dict],
        lr_scheduler: Union[DictConfig, Dict],
        criterion: Union[DictConfig, Dict],
        monitor: str = "val_loss",
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.network = network
        self._optimizer = OmegaConf.create(optimizer)
        self._lr_scheduler = OmegaConf.create(lr_scheduler)
        self.loss_fn = self.configure_criterion(criterion)

        # Accuracy metric
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    @staticmethod
    def configure_criterion(criterion: Union[DictConfig, Dict]) -> Type[nn.Module]:
        """Returns a loss functions."""
        criterion = OmegaConf.create(criterion)
        args = {} or criterion.args
        return getattr(nn, criterion.type)(**args)

    def _configure_optimizer(self) -> Type[torch.optim.Optimizer]:
        """Configures the optimizer."""
        args = {} or self._optimizer.args
        if self._optimizer.type == "MADGRAD":
            optimizer_class = madgrad.MADGRAD
        else:
            optimizer_class = getattr(torch.optim, self._optimizer.type)
        return optimizer_class(params=self.parameters(), **args)

    def _configure_lr_scheduler(self) -> Dict[str, Any]:
        """Configures the lr scheduler."""
        scheduler = {"monitor": self.monitor}
        args = {} or self._lr_scheduler.args

        if "interval" in args:
            scheduler["interval"] = args.pop("interval")

        scheduler["scheduler"] = getattr(
            torch.optim.lr_scheduler, self._lr_scheduler.type
        )(**args)
        return scheduler

    def configure_optimizers(self) -> Tuple[List[type], List[Dict[str, Any]]]:
        """Configures optimizer and lr scheduler."""
        optimizer = self._configure_optimizer()
        scheduler = self._configure_lr_scheduler()

        return [optimizer], [scheduler]

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

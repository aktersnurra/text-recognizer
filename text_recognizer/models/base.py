"""Base PyTorch Lightning model."""
from typing import Any, Dict, List, Tuple, Type

import attr
import hydra
from loguru import logger as log
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch import Tensor
import torchmetrics

from text_recognizer.networks.base import BaseNetwork


@attr.s
class BaseLitModel(LightningModule):
    """Abstract PyTorch Lightning class."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    network: Type[BaseNetwork] = attr.ib()
    criterion_config: DictConfig = attr.ib(converter=DictConfig)
    optimizer_config: DictConfig = attr.ib(converter=DictConfig)
    lr_scheduler_config: DictConfig = attr.ib(converter=DictConfig)

    interval: str = attr.ib()
    monitor: str = attr.ib(default="val/loss")

    loss_fn: Type[nn.Module] = attr.ib(init=False)

    train_acc: torchmetrics.Accuracy = attr.ib(
        init=False, default=torchmetrics.Accuracy()
    )
    val_acc: torchmetrics.Accuracy = attr.ib(
        init=False, default=torchmetrics.Accuracy()
    )
    test_acc: torchmetrics.Accuracy = attr.ib(
        init=False, default=torchmetrics.Accuracy()
    )

    @loss_fn.default
    def configure_criterion(self) -> Type[nn.Module]:
        """Returns a loss functions."""
        log.info(f"Instantiating criterion <{self.criterion_config._target_}>")
        return hydra.utils.instantiate(self.criterion_config)

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_idx: int,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _configure_optimizer(self) -> Type[torch.optim.Optimizer]:
        """Configures the optimizer."""
        log.info(f"Instantiating optimizer <{self.optimizer_config._target_}>")
        return hydra.utils.instantiate(self.optimizer_config, params=self.parameters())

    def _configure_lr_scheduler(
        self, optimizer: Type[torch.optim.Optimizer]
    ) -> Dict[str, Any]:
        """Configures the lr scheduler."""
        log.info(
            f"Instantiating learning rate scheduler <{self.lr_scheduler_config._target_}>"
        )
        scheduler = {
            "monitor": self.monitor,
            "interval": self.interval,
            "scheduler": hydra.utils.instantiate(
                self.lr_scheduler_config, optimizer=optimizer
            ),
        }
        return scheduler

    def configure_optimizers(self) -> Tuple[List[type], List[Dict[str, Any]]]:
        """Configures optimizer and lr scheduler."""
        optimizer = self._configure_optimizer()
        scheduler = self._configure_lr_scheduler(optimizer)
        return [optimizer], [scheduler]

    def forward(self, data: Tensor) -> Tensor:
        """Feedforward pass."""
        return self.network(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits = self(data)
        loss = self.loss_fn(logits, targets)
        self.log("train/loss", loss)
        self.train_acc(logits, targets)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, targets = batch
        logits = self(data)
        loss = self.loss_fn(logits, targets)
        self.log("val/loss", loss, prog_bar=True)
        self.val_acc(logits, targets)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch
        logits = self(data)
        self.test_acc(logits, targets)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

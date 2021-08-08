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

from text_recognizer.data.base_mapping import AbstractMapping


@attr.s(eq=False)
class BaseLitModel(LightningModule):
    """Abstract PyTorch Lightning class."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    network: Type[nn.Module] = attr.ib()
    mapping: Type[AbstractMapping] = attr.ib()
    loss_fn: Type[nn.Module] = attr.ib()
    optimizer_configs: DictConfig = attr.ib()
    lr_scheduler_configs: DictConfig = attr.ib()
    train_acc: torchmetrics.Accuracy = attr.ib(
        init=False, default=torchmetrics.Accuracy()
    )
    val_acc: torchmetrics.Accuracy = attr.ib(
        init=False, default=torchmetrics.Accuracy()
    )
    test_acc: torchmetrics.Accuracy = attr.ib(
        init=False, default=torchmetrics.Accuracy()
    )

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_idx: int,
    ) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _configure_optimizer(self) -> List[Type[torch.optim.Optimizer]]:
        """Configures the optimizer."""
        optimizers = []
        for optimizer_config in self.optimizer_configs.values():
            network = getattr(self, optimizer_config.parameters)
            del optimizer_config.parameters
            log.info(f"Instantiating optimizer <{optimizer_config._target_}>")
            optimizers.append(
                hydra.utils.instantiate(
                    self.optimizer_config, params=network.parameters()
                )
            )
        return optimizers

    def _configure_lr_schedulers(
        self, optimizers: List[Type[torch.optim.Optimizer]]
    ) -> List[Dict[str, Any]]:
        """Configures the lr scheduler."""
        schedulers = []
        for optimizer, lr_scheduler_config in zip(
            optimizers, self.lr_scheduler_configs.values()
        ):
            # Extract non-class arguments.
            monitor = lr_scheduler_config.monitor
            interval = lr_scheduler_config.interval
            del lr_scheduler_config.monitor
            del lr_scheduler_config.interval

            log.info(
                f"Instantiating learning rate scheduler <{lr_scheduler_config._target_}>"
            )
            scheduler = {
                "monitor": monitor,
                "interval": interval,
                "scheduler": hydra.utils.instantiate(
                    lr_scheduler_config, optimizer=optimizer
                ),
            }
            schedulers.append(scheduler)

        return schedulers

    def configure_optimizers(
        self,
    ) -> Tuple[List[Type[torch.optim.Optimizer]], List[Dict[str, Any]]]:
        """Configures optimizer and lr scheduler."""
        optimizers = self._configure_optimizer()
        schedulers = self._configure_lr_scheduler(optimizers)
        return optimizers, schedulers

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

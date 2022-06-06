"""Base PyTorch Lightning model."""
from typing import Any, Dict, List, Optional, Tuple, Type

import hydra
from loguru import logger as log
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch import Tensor
from torchmetrics import Accuracy

from text_recognizer.data.mappings.base import AbstractMapping


class LitBase(LightningModule):
    """Abstract PyTorch Lightning class."""

    def __init__(
        self,
        network: Type[nn.Module],
        loss_fn: Type[nn.Module],
        optimizer_configs: DictConfig,
        lr_scheduler_configs: Optional[DictConfig],
        mapping: Type[AbstractMapping],
    ) -> None:
        super().__init__()

        self.network = network
        self.loss_fn = loss_fn
        self.optimizer_configs = optimizer_configs
        self.lr_scheduler_configs = lr_scheduler_configs
        self.mapping = mapping

        # Placeholders
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_idx: int,
    ) -> None:
        """Optimal way to set grads to zero."""
        optimizer.zero_grad(set_to_none=True)

    def _configure_optimizer(self) -> List[Type[torch.optim.Optimizer]]:
        """Configures the optimizer."""
        optimizers = []
        for optimizer_config in self.optimizer_configs.values():
            module = self
            for m in str(optimizer_config.parameters).split("."):
                module = getattr(module, m)
            del optimizer_config.parameters
            log.info(f"Instantiating optimizer <{optimizer_config._target_}>")
            optimizers.append(
                hydra.utils.instantiate(optimizer_config, params=module.parameters())
            )
        return optimizers

    def _configure_lr_schedulers(
        self, optimizers: List[Type[torch.optim.Optimizer]]
    ) -> List[Dict[str, Any]]:
        """Configures the lr scheduler."""
        if self.lr_scheduler_configs is None:
            return []
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
        schedulers = self._configure_lr_schedulers(optimizers)
        return optimizers, schedulers

    def forward(self, data: Tensor) -> Tensor:
        """Feedforward pass."""
        return self.network(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        pass

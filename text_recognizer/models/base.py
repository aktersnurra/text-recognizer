"""Base PyTorch Lightning model."""
from typing import Any, Dict, Optional, Tuple, Type

import hydra
import torch
from loguru import logger as log
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torchmetrics import Accuracy

from text_recognizer.data.tokenizer import Tokenizer


class LitBase(LightningModule):
    """Abstract PyTorch Lightning class."""

    def __init__(
        self,
        network: Type[nn.Module],
        loss_fn: Type[nn.Module],
        optimizer_config: DictConfig,
        lr_scheduler_config: Optional[DictConfig],
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__()

        self.network = network
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.tokenizer = tokenizer
        ignore_index = int(self.tokenizer.get_value("<p>"))
        # Placeholders
        self.train_acc = Accuracy(mdmc_reduce="samplewise", ignore_index=ignore_index)
        self.val_acc = Accuracy(mdmc_reduce="samplewise", ignore_index=ignore_index)
        self.test_acc = Accuracy(mdmc_reduce="samplewise", ignore_index=ignore_index)

    def optimizer_zero_grad(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_idx: int,
    ) -> None:
        """Optimal way to set grads to zero."""
        optimizer.zero_grad(set_to_none=True)

    def _configure_optimizer(self) -> Type[torch.optim.Optimizer]:
        """Configures the optimizer."""
        log.info(f"Instantiating optimizer <{self.optimizer_config._target_}>")
        return hydra.utils.instantiate(
            self.optimizer_config, params=self.network.parameters()
        )

    def _configure_lr_schedulers(
        self, optimizer: Type[torch.optim.Optimizer]
    ) -> Dict[str, Any]:
        """Configures the lr scheduler."""
        log.info(
            f"Instantiating learning rate scheduler <{self.lr_scheduler_config._target_}>"
        )
        monitor = self.lr_scheduler_config.monitor
        interval = self.lr_scheduler_config.interval
        del self.lr_scheduler_config.monitor
        del self.lr_scheduler_config.interval

        return {
            "monitor": monitor,
            "interval": interval,
            "scheduler": hydra.utils.instantiate(
                self.lr_scheduler_config, optimizer=optimizer
            ),
        }

    def configure_optimizers(
        self,
    ) -> Dict[str, Any]:
        """Configures optimizer and lr scheduler."""
        optimizer = self._configure_optimizer()
        scheduler = self._configure_lr_schedulers(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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

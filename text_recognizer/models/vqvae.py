"""PyTorch Lightning model for base Transformers."""
from typing import Any, Dict, Union, Tuple, Type

import attr
from omegaconf import DictConfig
from torch import nn
from torch import Tensor
import wandb

from text_recognizer.models.base import BaseLitModel


@attr.s(auto_attribs=True)
class VQVAELitModel(BaseLitModel):
    """A PyTorch Lightning model for transformer networks."""

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.network.predict(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, _ = batch
        reconstructions, vq_loss = self.network(data)
        loss = self.loss_fn(reconstructions, data)
        loss += vq_loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, _ = batch
        reconstructions, vq_loss = self.network(data)
        loss = self.loss_fn(reconstructions, data)
        loss += vq_loss
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, _ = batch
        reconstructions, vq_loss = self.network(data)
        loss = self.loss_fn(reconstructions, data)
        loss += vq_loss
        self.log("test/loss", loss)

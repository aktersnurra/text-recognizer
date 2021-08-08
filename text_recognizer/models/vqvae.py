"""PyTorch Lightning model for base Transformers."""
from typing import Tuple

import attr
from torch import Tensor

from text_recognizer.models.base import BaseLitModel


@attr.s(auto_attribs=True, eq=False)
class VQVAELitModel(BaseLitModel):
    """A PyTorch Lightning model for transformer networks."""

    latent_loss_weight: float = attr.ib(default=0.25)

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.network(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, _ = batch

        reconstructions, vq_loss = self(data)
        loss = self.loss_fn(reconstructions, data)
        loss = loss + self.latent_loss_weight * vq_loss

        self.log("train/vq_loss", vq_loss)
        self.log("train/loss", loss)

        # self.train_acc(reconstructions, data)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, _ = batch
        reconstructions, vq_loss = self(data)
        loss = self.loss_fn(reconstructions, data)
        loss = loss + self.latent_loss_weight * vq_loss

        self.log("val/vq_loss", vq_loss)
        self.log("val/loss", loss, prog_bar=True)

        # self.val_acc(reconstructions, data)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, _ = batch
        reconstructions, vq_loss = self(data)
        loss = self.loss_fn(reconstructions, data)
        loss = loss + self.latent_loss_weight * vq_loss
        self.log("test/vq_loss", vq_loss)
        self.log("test/loss", loss)
        # self.test_acc(reconstructions, data)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

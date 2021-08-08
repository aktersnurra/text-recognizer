"""PyTorch Lightning model for base Transformers."""
from typing import Tuple

import attr
from torch import Tensor

from text_recognizer.models.base import BaseLitModel
from text_recognizer.criterions.vqgan_loss import VQGANLoss


@attr.s(auto_attribs=True, eq=False)
class VQVAELitModel(BaseLitModel):
    """A PyTorch Lightning model for transformer networks."""

    loss_fn: VQGANLoss = attr.ib()
    latent_loss_weight: float = attr.ib(default=0.25)

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.network(data)

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int
    ) -> Tensor:
        """Training step."""
        data, _ = batch

        reconstructions, vq_loss = self(data)
        loss = self.loss_fn(reconstructions, data)

        if optimizer_idx == 0:
            loss, log = self.loss_fn(
                data=data,
                reconstructions=reconstructions,
                vq_loss=vq_loss,
                optimizer_idx=optimizer_idx,
                stage="train",
            )
            self.log(
                "train/loss",
                loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(log, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return loss

        if optimizer_idx == 1:
            loss, log = self.loss_fn(
                data=data,
                reconstructions=reconstructions,
                vq_loss=vq_loss,
                optimizer_idx=optimizer_idx,
                stage="train",
            )
            self.log(
                "train/discriminator_loss",
                loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(log, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, _ = batch
        reconstructions, vq_loss = self(data)

        loss, log = self.loss_fn(
            data=data,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=0,
            stage="val",
        )
        self.log(
            "val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log(
            "val/rec_loss",
            log["val/rec_loss"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(log)

        _, log = self.loss_fn(
            data=data,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=1,
            stage="val",
        )
        self.log_dict(log)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, _ = batch
        reconstructions, vq_loss = self(data)

        loss, log = self.loss_fn(
            data=data,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=0,
            stage="test",
        )
        self.log(
            "test/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log(
            "test/rec_loss",
            log["test/rec_loss"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(log)

        _, log = self.loss_fn(
            data=data,
            reconstructions=reconstructions,
            vq_loss=vq_loss,
            optimizer_idx=1,
            stage="test",
        )
        self.log_dict(log)

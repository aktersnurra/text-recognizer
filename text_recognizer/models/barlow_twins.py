"""PyTorch Lightning Barlow Twins model."""
from typing import Tuple, Type
import attr
from torch import nn
from torch import Tensor

from text_recognizer.models.base import BaseLitModel
from text_recognizer.criterions.barlow_twins import BarlowTwinsLoss


@attr.s(auto_attribs=True, eq=False)
class BarlowTwinsLitModel(BaseLitModel):
    """Barlow Twins training proceduer."""

    network: Type[nn.Module] = attr.ib()
    loss_fn: BarlowTwinsLoss = attr.ib()

    def forward(self, data: Tensor) -> Tensor:
        """Encodes image to projector latent."""
        return self.network(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, _ = batch
        x1, x2 = data
        z1, z2 = self(x1), self(x2)
        loss = self.loss_fn(z1, z2)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, _ = batch
        x1, x2 = data
        z1, z2 = self(x1), self(x2)
        loss = self.loss_fn(z1, z2)
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, _ = batch
        x1, x2 = data
        z1, z2 = self(x1), self(x2)
        loss = self.loss_fn(z1, z2)
        self.log("test/loss", loss, prog_bar=True)

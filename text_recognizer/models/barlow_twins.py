"""PyTorch Lightning Barlow Twins model."""
from typing import Type
import attr
import pytorch_lightning as pl
import torch
from torch import nn
from torch import Tensor
import torchvision.transforms as T

from text_recognizer.models.base import BaseLitModel
from text_recognizer.networks.barlow_twins.projector import Projector


def off_diagonal(x: Tensor) -> Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@attr.s(auto_attribs=True, eq=False)
class BarlowTwinsLitModel(BaseLitModel):
    """Barlow Twins training proceduer."""

    encoder: Type[nn.Module] = attr.ib()
    projector: Projector = attr.ib()
    lambda_: float = attr.ib()
    augment: T.Compose = attr.ib()

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        self.bn = nn.BatchNorm1d(self.projector.dims[-1], affine=False)

    def loss_fn(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Calculates the Barlow Twin loss."""
        c = torch.mm(self.bn(z1), self.bn(z2))
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambda_ * off_diag

    def forward(self, data: Tensor) -> Tensor:
        """Encodes image to projector latent."""
        z_e = self.encoder(data).flatten(start_dim=1)
        z_p = self.projector(z_e)
        return z_p

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, _ = batch
        x1, x2 = self.augment(data), self.augment(data)
        z1, z2 = self(x1), self(x2)
        loss = self.loss_fn(z1, z2)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, _ = batch
        x1, x2 = self.augment(data), self.augment(data)
        z1, z2 = self(x1), self(x2)
        loss = self.loss_fn(z1, z2)
        self.log("val/loss", loss, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, _ = batch
        x1, x2 = self.augment(data), self.augment(data)
        z1, z2 = self(x1), self(x2)
        loss = self.loss_fn(z1, z2)
        self.log("test/loss", loss, prog_bar=True)

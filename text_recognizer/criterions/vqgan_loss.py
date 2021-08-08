"""VQGAN loss for PyTorch Lightning."""
from typing import Dict
from click.types import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from text_recognizer.criterions.n_layer_discriminator import NLayerDiscriminator


class VQGANLoss(nn.Module):
    """VQGAN loss."""

    def __init__(
        self,
        reconstruction_loss: nn.L1Loss,
        discriminator: NLayerDiscriminator,
        vq_loss_weight: float = 1.0,
        discriminator_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.discriminator = discriminator
        self.vq_loss_weight = vq_loss_weight
        self.discriminator_weight = discriminator_weight

    @staticmethod
    def adversarial_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
        """Calculates the adversarial loss."""
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        d_loss = (loss_real + loss_fake) / 2.0
        return d_loss

    def forward(
        self,
        data: Tensor,
        reconstructions: Tensor,
        vq_loss: Tensor,
        optimizer_idx: int,
        stage: str,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Calculates the VQGAN loss."""
        rec_loss = self.reconstruction_loss(
            data.contiguous(), reconstructions.contiguous()
        )

        # GAN part.
        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            loss = (
                rec_loss
                + self.discriminator_weight * g_loss
                + self.vq_loss_weight * vq_loss
            )
            log = {
                f"{stage}/loss": loss,
                f"{stage}/vq_loss": vq_loss,
                f"{stage}/rec_loss": rec_loss,
                f"{stage}/g_loss": g_loss,
            }
            return loss, log

        if optimizer_idx == 1:
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            logits_real = self.discriminator(data.contiguous().detach())

            d_loss = self.adversarial_loss(
                logits_real=logits_real, logits_fake=logits_fake
            )
            loss = (
                rec_loss
                + self.discriminator_weight * d_loss
                + self.vq_loss_weight * vq_loss
            )
            log = {
                f"{stage}/loss": loss,
                f"{stage}/vq_loss": vq_loss,
                f"{stage}/rec_loss": rec_loss,
                f"{stage}/d_loss": d_loss,
            }
            return loss, log

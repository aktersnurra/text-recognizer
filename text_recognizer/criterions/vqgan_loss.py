"""VQGAN loss for PyTorch Lightning."""
from typing import Dict, Optional
from click.types import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from text_recognizer.criterions.n_layer_discriminator import NLayerDiscriminator


def adopt_weight(
    weight: Tensor, global_step: int, threshold: int = 0, value: float = 0.0
) -> float:
    if global_step < threshold:
        weight = value
    return weight


class VQGANLoss(nn.Module):
    """VQGAN loss."""

    def __init__(
        self,
        reconstruction_loss: nn.L1Loss,
        discriminator: NLayerDiscriminator,
        vq_loss_weight: float = 1.0,
        discriminator_weight: float = 1.0,
        discriminator_factor: float = 1.0,
        discriminator_iter_start: int = 1000,
    ) -> None:
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.discriminator = discriminator
        self.vq_loss_weight = vq_loss_weight
        self.discriminator_weight = discriminator_weight
        self.discriminator_factor = discriminator_factor
        self.discriminator_iter_start = discriminator_iter_start

    @staticmethod
    def adversarial_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
        """Calculates the adversarial loss."""
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        d_loss = (loss_real + loss_fake) / 2.0
        return d_loss

    def _adaptive_weight(
        self, rec_loss: Tensor, g_loss: Tensor, decoder_last_layer: Tensor
    ) -> Tensor:
        rec_grads = torch.autograd.grad(
            rec_loss, decoder_last_layer, retain_graph=True
        )[0]
        g_grads = torch.autograd.grad(g_loss, decoder_last_layer, retain_graph=True)[0]
        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1.0e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1.0e4).detach()
        d_weight *= self.discriminator_weight
        return d_weight

    def forward(
        self,
        data: Tensor,
        reconstructions: Tensor,
        vq_loss: Tensor,
        decoder_last_layer: Tensor,
        optimizer_idx: int,
        global_step: int,
        stage: str,
    ) -> Optional[Tuple]:
        """Calculates the VQGAN loss."""
        rec_loss: Tensor = self.reconstruction_loss(
            data.contiguous(), reconstructions.contiguous()
        )

        # GAN part.
        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            if self.training:
                d_weight = self._adaptive_weight(
                    rec_loss=rec_loss,
                    g_loss=g_loss,
                    decoder_last_layer=decoder_last_layer,
                )
            else:
                d_weight = torch.tensor(0.0)

            d_factor = adopt_weight(
                self.discriminator_factor,
                global_step=global_step,
                threshold=self.discriminator_iter_start,
            )

            loss: Tensor = (
                rec_loss + d_factor * d_weight * g_loss + self.vq_loss_weight * vq_loss
            )
            log = {
                f"{stage}/total_loss": loss,
                f"{stage}/vq_loss": vq_loss,
                f"{stage}/rec_loss": rec_loss,
                f"{stage}/g_loss": g_loss,
            }
            return loss, log

        if optimizer_idx == 1:
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            logits_real = self.discriminator(data.contiguous().detach())

            d_factor = adopt_weight(
                self.discriminator_factor,
                global_step=global_step,
                threshold=self.discriminator_iter_start,
            )

            d_loss = d_factor * self.adversarial_loss(
                logits_real=logits_real, logits_fake=logits_fake
            )

            log = {
                f"{stage}/d_loss": d_loss,
            }
            return d_loss, log

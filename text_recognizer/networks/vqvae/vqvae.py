"""The VQ-VAE."""
from typing import Tuple

from torch import nn
from torch import Tensor

from text_recognizer.networks.quantizer.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """Vector Quantized Variational AutoEncoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: VectorQuantizer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

    def encode(self, x: Tensor) -> Tensor:
        """Encodes input to a latent code."""
        return self.encoder(x)

    def quantize(self, z_e: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantizes the encoded latent vectors."""
        z_q, _, commitment_loss = self.quantizer(z_e)
        return z_q, commitment_loss

    def decode(self, z_q: Tensor) -> Tensor:
        """Reconstructs input from latent codes."""
        return self.decoder(z_q)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compresses and decompresses input."""
        z_e = self.encode(x)
        z_q, commitment_loss = self.quantize(z_e)
        x_hat = self.decode(z_q)
        return x_hat, commitment_loss

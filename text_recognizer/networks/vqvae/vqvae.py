"""The VQ-VAE."""
from typing import Tuple

from torch import nn
from torch import Tensor

from text_recognizer.networks.vqvae.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """Vector Quantized Variational AutoEncoder."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        hidden_dim: int,
        embedding_dim: int,
        num_embeddings: int,
        decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pre_codebook_conv = nn.Conv2d(
            in_channels=hidden_dim, out_channels=embedding_dim, kernel_size=1
        )
        self.post_codebook_conv = nn.Conv2d(
            in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=1
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, decay=decay,
        )


    def encode(self, x: Tensor) -> Tensor:
        """Encodes input to a latent code."""
        z_e = self.encoder(x)
        return self.pre_codebook_conv(z_e)

    def quantize(self, z_e: Tensor) -> Tuple[Tensor, Tensor]:
        z_q, vq_loss = self.quantizer(z_e)
        return z_q, vq_loss

    def decode(self, z_q: Tensor) -> Tensor:
        """Reconstructs input from latent codes."""
        z = self.post_codebook_conv(z_q)
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compresses and decompresses input."""
        z_e = self.encode(x)
        z_q, vq_loss = self.quantize(z_e)
        x_hat = self.decode(z_q)
        return x_hat, vq_loss

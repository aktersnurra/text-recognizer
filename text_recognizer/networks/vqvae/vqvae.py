"""The VQ-VAE."""
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from text_recognizer.networks.vqvae.decoder import Decoder
from text_recognizer.networks.vqvae.encoder import Encoder
from text_recognizer.networks.vqvae.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """Vector Quantized Variational AutoEncoder."""

    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        num_residual_layers: int,
        embedding_dim: int,
        num_embeddings: int,
        decay: float = 0.99,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        # Encoders
        self.btm_encoder = Encoder(
            in_channels=1,
            out_channels=embedding_dim,
            res_channels=res_channels,
            num_residual_layers=num_residual_layers,
            embedding_dim=embedding_dim,
            activation=activation,
        )

        self.top_encoder = Encoder(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            res_channels=res_channels,
            num_residual_layers=num_residual_layers,
            embedding_dim=embedding_dim,
            activation=activation,
        )

        # Quantizers
        self.btm_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, decay=decay,
        )

        self.top_quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, decay=decay,
        )

        # Decoders
        self.top_decoder = Decoder(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            embedding_dim=embedding_dim,
            res_channels=res_channels,
            num_residual_layers=num_residual_layers,
            activation=activation,
        )

        self.btm_decoder = Decoder(
            in_channels=2 * embedding_dim,
            out_channels=in_channels,
            embedding_dim=embedding_dim,
            res_channels=res_channels,
            num_residual_layers=num_residual_layers,
            activation=activation,
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes input to a latent code."""
        z_btm = self.btm_encoder(x)
        z_top = self.top_encoder(z_btm)
        return z_btm, z_top

    def quantize(
        self, z_btm: Tensor, z_top: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        q_btm, vq_btm_loss = self.top_quantizer(z_btm)
        q_top, vq_top_loss = self.top_quantizer(z_top)
        return q_btm, vq_btm_loss, q_top, vq_top_loss

    def decode(self, q_btm: Tensor, q_top: Tensor) -> Tuple[Tensor, Tensor]:
        """Reconstructs input from latent codes."""
        d_top = self.top_decoder(q_top)
        x_hat = self.btm_decoder(torch.cat((d_top, q_btm), dim=1))
        return d_top, x_hat

    def loss_fn(
        self, vq_btm_loss: Tensor, vq_top_loss: Tensor, d_top: Tensor, z_btm: Tensor
    ) -> Tensor:
        """Calculates the latent loss."""
        return 0.5 * (vq_top_loss + vq_btm_loss) + F.mse_loss(d_top, z_btm)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compresses and decompresses input."""
        z_btm, z_top = self.encode(x)
        q_btm, vq_btm_loss, q_top, vq_top_loss = self.quantize(z_btm=z_btm, z_top=z_top)
        d_top, x_hat = self.decode(q_btm=q_btm, q_top=q_top)
        vq_loss = self.loss_fn(
            vq_btm_loss=vq_btm_loss, vq_top_loss=vq_top_loss, d_top=d_top, z_btm=z_btm
        )
        return x_hat, vq_loss

"""The VQ-VAE."""
from typing import Any, Dict, List, Optional, Tuple

from torch import nn
from torch import Tensor

from text_recognizer.networks.vqvae import Decoder, Encoder


class VQVAE(nn.Module):
    """Vector Quantized Variational AutoEncoder."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        num_residual_layers: int,
        embedding_dim: int,
        num_embeddings: int,
        upsampling: Optional[List[List[int]]] = None,
        beta: float = 0.25,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        *args: Any,
        **kwargs: Dict,
    ) -> None:
        super().__init__()

        # configure encoder.
        self.encoder = Encoder(
            in_channels,
            channels,
            kernel_sizes,
            strides,
            num_residual_layers,
            embedding_dim,
            num_embeddings,
            beta,
            activation,
            dropout_rate,
        )

        # Configure decoder.
        channels.reverse()
        kernel_sizes.reverse()
        strides.reverse()
        self.decoder = Decoder(
            channels,
            kernel_sizes,
            strides,
            num_residual_layers,
            embedding_dim,
            upsampling,
            activation,
            dropout_rate,
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes input to a latent code."""
        return self.encoder(x)

    def decode(self, z_q: Tensor) -> Tensor:
        """Reconstructs input from latent codes."""
        return self.decoder(z_q)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compresses and decompresses input."""
        if len(x.shape) < 4:
            x = x[(None,) * (4 - len(x.shape))]
        z_q, vq_loss = self.encode(x)
        x_reconstruction = self.decode(z_q)
        return x_reconstruction, vq_loss

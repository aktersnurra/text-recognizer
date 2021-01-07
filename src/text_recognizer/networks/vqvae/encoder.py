"""CNN encoder for the VQ-VAE."""

from typing import List, Optional, Type

import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function
from text_recognizer.networks.vqvae.vector_quantizer import VectorQuantizer


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Type[nn.Module],
        dropout: Optional[Type[nn.Module]],
    ) -> None:
        super().__init__()
        self.block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        ]

        if dropout is not None:
            self.block.append(dropout)

        self.block = nn.Sequential(*self.block)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual forward pass."""
        return x + self.block(x)


class Encoder(nn.Module):
    """A CNN encoder network."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        num_residual_layers: int,
        embedding_dim: int,
        num_embeddings: int,
        beta: float = 0.25,
        activation: str = "elu",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        pass
        # if dropout_rate:
        #   if activation == "selu":
        #      dropout = nn.AlphaDropout(p=dropout_rate)
        # else:
        #       dropout = nn.Dropout(p=dropout_rate)
        # else:
        #   dropout = None

    def _build_encoder(self) -> nn.Sequential:
        # TODO: Continue to implement encoder.
        pass

"""CNN encoder for the VQ-VAE."""
from typing import List, Optional, Tuple, Type

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
        dropout: Optional[Type[nn.Module]],
    ) -> None:
        super().__init__()
        self.block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
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
        kernel_sizes: List[int],
        strides: List[int],
        num_residual_layers: int,
        embedding_dim: int,
        num_embeddings: int,
        beta: float = 0.25,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if dropout_rate:
            if activation == "selu":
                dropout = nn.AlphaDropout(p=dropout_rate)
            else:
                dropout = nn.Dropout(p=dropout_rate)
        else:
            dropout = None

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        activation = activation_function(activation)

        # Configure encoder.
        self.encoder = self._build_encoder(
            in_channels,
            channels,
            kernel_sizes,
            strides,
            num_residual_layers,
            activation,
            dropout,
        )

        # Configure Vector Quantizer.
        self.vector_quantizer = VectorQuantizer(
            self.num_embeddings, self.embedding_dim, self.beta
        )

    def _build_compression_block(
        self,
        in_channels: int,
        channels: int,
        kernel_sizes: List[int],
        strides: List[int],
        activation: Type[nn.Module],
        dropout: Optional[Type[nn.Module]],
    ) -> nn.ModuleList:
        modules = nn.ModuleList([])
        configuration = zip(channels, kernel_sizes, strides)
        for out_channels, kernel_size, stride in configuration:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size, stride=stride, padding=1
                    ),
                    activation,
                )
            )

            if dropout is not None:
                modules.append(dropout)

            in_channels = out_channels

        return modules

    def _build_encoder(
        self,
        in_channels: int,
        channels: int,
        kernel_sizes: List[int],
        strides: List[int],
        num_residual_layers: int,
        activation: Type[nn.Module],
        dropout: Optional[Type[nn.Module]],
    ) -> nn.Sequential:
        encoder = nn.ModuleList([])

        # compression module
        encoder.extend(
            self._build_compression_block(
                in_channels, channels, kernel_sizes, strides, activation, dropout
            )
        )

        # Bottleneck module.
        encoder.extend(
            nn.ModuleList(
                [
                    _ResidualBlock(channels[-1], channels[-1], dropout)
                    for i in range(num_residual_layers)
                ]
            )
        )

        encoder.append(
            nn.Conv2d(
                channels[-1],
                self.embedding_dim,
                kernel_size=1,
                stride=1,
            )
        )

        return nn.Sequential(*encoder)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes input into a discrete representation."""
        z_e = self.encoder(x)
        z_q, vq_loss = self.vector_quantizer(z_e)
        return z_q, vq_loss

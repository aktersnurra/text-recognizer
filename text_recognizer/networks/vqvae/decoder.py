"""CNN decoder for the VQ-VAE."""

from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function
from text_recognizer.networks.vqvae.encoder import _ResidualBlock


class Decoder(nn.Module):
    """A CNN encoder network."""

    def __init__(
        self,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        num_residual_layers: int,
        embedding_dim: int,
        upsampling: Optional[List[List[int]]] = None,
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

        self.upsampling = upsampling

        self.res_block = nn.ModuleList([])
        self.upsampling_block = nn.ModuleList([])

        self.embedding_dim = embedding_dim
        activation = activation_function(activation)

        # Configure encoder.
        self.decoder = self._build_decoder(
            channels, kernel_sizes, strides, num_residual_layers, activation, dropout,
        )

    def _build_decompression_block(
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
        for i, (out_channels, kernel_size, stride) in enumerate(configuration):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=1,
                    ),
                    activation,
                )
            )

            if i < len(self.upsampling):
                modules.append(nn.Upsample(size=self.upsampling[i]),)

            if dropout is not None:
                modules.append(dropout)

            in_channels = out_channels

        modules.extend(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, 1, kernel_size=kernel_size, stride=stride, padding=1
                ),
                nn.Tanh(),
            )
        )

        return modules

    def _build_decoder(
        self,
        channels: int,
        kernel_sizes: List[int],
        strides: List[int],
        num_residual_layers: int,
        activation: Type[nn.Module],
        dropout: Optional[Type[nn.Module]],
    ) -> nn.Sequential:

        self.res_block.append(
            nn.Conv2d(self.embedding_dim, channels[0], kernel_size=1, stride=1,)
        )

        # Bottleneck module.
        self.res_block.extend(
            nn.ModuleList(
                [
                    _ResidualBlock(channels[0], channels[0], dropout)
                    for i in range(num_residual_layers)
                ]
            )
        )

        # Decompression module
        self.upsampling_block.extend(
            self._build_decompression_block(
                channels[0], channels[1:], kernel_sizes, strides, activation, dropout
            )
        )

        self.res_block = nn.Sequential(*self.res_block)
        self.upsampling_block = nn.Sequential(*self.upsampling_block)

        return nn.Sequential(self.res_block, self.upsampling_block)

    def forward(self, z_q: Tensor) -> Tensor:
        """Reconstruct input from given codes."""
        x_reconstruction = self.decoder(z_q)
        return x_reconstruction

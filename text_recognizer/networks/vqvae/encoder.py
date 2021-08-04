"""CNN encoder for the VQ-VAE."""
from typing import Sequence, Optional, Tuple, Type

import attr
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function
from text_recognizer.networks.vqvae.residual import Residual


@attr.s(eq=False)
class Encoder(nn.Module):
    """A CNN encoder network."""

    in_channels: int = attr.ib()
    out_channels: int = attr.ib()
    res_channels: int = attr.ib()
    num_residual_layers: int = attr.ib()
    embedding_dim: int = attr.ib()
    activation: str = attr.ib()
    encoder: nn.Sequential = attr.ib(init=False)

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        self.encoder = self._build_compression_block()

    def _build_compression_block(self) -> nn.Sequential:
        activation_fn = activation_function(self.activation)
        block = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation_fn,
            nn.Conv2d(
                in_channels=self.out_channels // 2,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation_fn,
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1,
            ),
        ]

        for _ in range(self.num_residual_layers):
            block.append(
                Residual(in_channels=self.out_channels, out_channels=self.res_channels)
            )

        block.append(
            nn.Conv2d(
                in_channels=self.out_channels,
                out_channels=self.embedding_dim,
                kernel_size=1,
            )
        )

        return nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes input into a discrete representation."""
        return self.encoder(x)

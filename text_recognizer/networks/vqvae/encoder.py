"""CNN encoder for the VQ-VAE."""
from typing import List, Tuple

from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function
from text_recognizer.networks.vqvae.residual import Residual


class Encoder(nn.Module):
    """A CNN encoder network."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        channels_multipliers: List[int],
        dropout_rate: float,
        activation: str = "mish",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.channels_multipliers = tuple(channels_multipliers)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.encoder = self._build_compression_block()

    def _build_compression_block(self) -> nn.Sequential:
        """Builds encoder network."""
        encoder = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        ]

        num_blocks = len(self.channels_multipliers)
        channels_multipliers = (1,) + self.channels_multipliers
        activation_fn = activation_function(self.activation)

        for i in range(num_blocks):
            in_channels = self.hidden_dim * channels_multipliers[i]
            out_channels = self.hidden_dim * channels_multipliers[i + 1]
            encoder += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                activation_fn,
            ]

        for _ in range(2):
            encoder += [
                Residual(
                    in_channels=self.hidden_dim * self.channels_multipliers[-1],
                    out_channels=self.hidden_dim * self.channels_multipliers[-1],
                    dropout_rate=self.dropout_rate,
                    use_norm=True,
                )
            ]

        return nn.Sequential(*encoder)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes input into a discrete representation."""
        return self.encoder(x)

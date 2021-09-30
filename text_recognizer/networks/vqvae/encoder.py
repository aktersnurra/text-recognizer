"""CNN encoder for the VQ-VAE."""
from typing import List, Tuple

from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function
from text_recognizer.networks.vqvae.norm import Normalize
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
        use_norm: bool = False,
        num_residuals: int = 4,
        residual_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.channels_multipliers = tuple(channels_multipliers)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.num_residuals = num_residuals
        self.residual_channels = residual_channels
        self.encoder = self._build_compression_block()

    def _build_compression_block(self) -> nn.Sequential:
        """Builds encoder network."""
        num_blocks = len(self.channels_multipliers)
        channels_multipliers = (1,) + self.channels_multipliers
        activation_fn = activation_function(self.activation)

        encoder = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        ]

        for i in range(num_blocks):
            in_channels = self.hidden_dim * channels_multipliers[i]
            out_channels = self.hidden_dim * channels_multipliers[i + 1]
            if self.use_norm:
                encoder += [
                    Normalize(num_channels=in_channels,),
                ]
            encoder += [
                activation_fn,
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            ]

        for _ in range(self.num_residuals):
            encoder += [
                Residual(
                    in_channels=out_channels,
                    residual_channels=self.residual_channels,
                    use_norm=self.use_norm,
                    activation=self.activation,
                )
            ]

        return nn.Sequential(*encoder)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes input into a discrete representation."""
        return self.encoder(x)

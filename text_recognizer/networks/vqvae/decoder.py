"""CNN decoder for the VQ-VAE."""
from typing import Sequence

from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function
from text_recognizer.networks.vqvae.norm import Normalize
from text_recognizer.networks.vqvae.residual import Residual


class Decoder(nn.Module):
    """A CNN encoder network."""

    def __init__(
        self,
        out_channels: int,
        hidden_dim: int,
        channels_multipliers: Sequence[int],
        dropout_rate: float,
        activation: str = "mish",
        use_norm: bool = False,
        num_residuals: int = 4,
        residual_channels: int = 32,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.channels_multipliers = tuple(channels_multipliers)
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        self.num_residuals = num_residuals
        self.residual_channels = residual_channels
        self.decoder = self._build_decompression_block()

    def _build_decompression_block(self,) -> nn.Sequential:
        decoder = []
        in_channels = self.hidden_dim * self.channels_multipliers[0]
        for _ in range(self.num_residuals):
            decoder += [
                Residual(
                    in_channels=in_channels,
                    residual_channels=self.residual_channels,
                    use_norm=self.use_norm,
                    activation=self.activation,
                ),
            ]

        activation_fn = activation_function(self.activation)
        out_channels_multipliers = self.channels_multipliers + (1,)
        num_blocks = len(self.channels_multipliers)

        for i in range(num_blocks):
            in_channels = self.hidden_dim * self.channels_multipliers[i]
            out_channels = self.hidden_dim * out_channels_multipliers[i + 1]
            if self.use_norm:
                decoder += [
                    Normalize(num_channels=in_channels,),
                ]
            decoder += [
                activation_fn,
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            ]

        if self.use_norm:
            decoder += [
                Normalize(
                    num_channels=self.hidden_dim * out_channels_multipliers[-1],
                    num_groups=self.hidden_dim * out_channels_multipliers[-1] // 4,
                ),
            ]

        decoder += [
            nn.Conv2d(
                in_channels=self.hidden_dim * out_channels_multipliers[-1],
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        ]
        return nn.Sequential(*decoder)

    def forward(self, z_q: Tensor) -> Tensor:
        """Reconstruct input from given codes."""
        return self.decoder(z_q)

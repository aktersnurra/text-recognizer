"""PixelCNN encoder and decoder.

Same as in VQGAN among other. Hopefully, better reconstructions...

TODO: Add num of residual layers.
"""
from typing import Sequence

from torch import nn
from torch import Tensor

from text_recognizer.networks.vqvae.attention import Attention
from text_recognizer.networks.vqvae.norm import Normalize
from text_recognizer.networks.vqvae.residual import Residual
from text_recognizer.networks.vqvae.resize import Downsample, Upsample


class Encoder(nn.Module):
    """PixelCNN encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        channels_multipliers: Sequence[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.channels_multipliers = tuple(channels_multipliers)
        self.encoder = self._build_encoder()

    def _build_encoder(self) -> nn.Sequential:
        """Builds encoder."""
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
        in_channels_multipliers = (1,) + self.channels_multipliers 
        for i in range(num_blocks):
            in_channels = self.hidden_dim * in_channels_multipliers[i]
            out_channels = self.hidden_dim * self.channels_multipliers[i]
            encoder += [
                Residual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=self.dropout_rate,
                    use_norm=True,
                ),
            ]
            if i == num_blocks - 1:
                encoder.append(Attention(in_channels=out_channels))
            encoder.append(Downsample())

        for _ in range(2):
            encoder += [
                Residual(
                    in_channels=self.hidden_dim * self.channels_multipliers[-1],
                    out_channels=self.hidden_dim * self.channels_multipliers[-1],
                    dropout_rate=self.dropout_rate,
                    use_norm=True,
                ),
                Attention(in_channels=self.hidden_dim * self.channels_multipliers[-1])
            ]

        encoder += [
            Normalize(num_channels=self.hidden_dim * self.channels_multipliers[-1]),
            nn.Mish(),
            nn.Conv2d(
                in_channels=self.hidden_dim * self.channels_multipliers[-1],
                out_channels=self.hidden_dim * self.channels_multipliers[-1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        ]
        return nn.Sequential(*encoder)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes input to a latent representation."""
        return self.encoder(x)


class Decoder(nn.Module):
    """PixelCNN decoder."""

    def __init__(
        self,
        hidden_dim: int,
        channels_multipliers: Sequence[int],
        out_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.channels_multipliers = tuple(channels_multipliers)
        self.dropout_rate = dropout_rate
        self.decoder = self._build_decoder()

    def _build_decoder(self) -> nn.Sequential:
        """Builds decoder."""
        in_channels = self.hidden_dim * self.channels_multipliers[0]
        decoder = [
            Residual(
                in_channels=in_channels,
                out_channels=in_channels,
                dropout_rate=self.dropout_rate,
                use_norm=True,
            ),
            Attention(in_channels=in_channels),
            Residual(
                in_channels=in_channels,
                out_channels=in_channels,
                dropout_rate=self.dropout_rate,
                use_norm=True,
            ),
        ]

        out_channels_multipliers = self.channels_multipliers + (1, )
        num_blocks = len(self.channels_multipliers)

        for i in range(num_blocks):
            in_channels = self.hidden_dim * self.channels_multipliers[i]
            out_channels = self.hidden_dim * out_channels_multipliers[i + 1]
            decoder.append(
                Residual(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=self.dropout_rate,
                    use_norm=True,
                )
            )
            if i == 0:
                decoder.append(
                    Attention(
                        in_channels=out_channels
                    )
                )
            decoder.append(Upsample())

        decoder += [
            Normalize(num_channels=self.hidden_dim * out_channels_multipliers[-1]),
            nn.Mish(),
            nn.Conv2d(
                in_channels=self.hidden_dim * out_channels_multipliers[-1],
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        ]
        return nn.Sequential(*decoder)

    def forward(self, x: Tensor) -> Tensor:
        """Decodes latent vector."""
        return self.decoder(x)

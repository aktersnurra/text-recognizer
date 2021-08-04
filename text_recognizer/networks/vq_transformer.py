"""Vector quantized encoder, transformer decoder."""
import math
from typing import Tuple

from torch import nn, Tensor

from text_recognizer.networks.encoders.efficientnet import EfficientNet
from text_recognizer.networks.conv_transformer import ConvTransformer
from text_recognizer.networks.transformer.layers import Decoder
from text_recognizer.networks.transformer.positional_encodings import (
    PositionalEncoding,
    PositionalEncoding2D,
)


class VqTransformer(ConvTransformer):
    """Convolutional encoder and transformer decoder network."""

    def __init__(
        self,
        input_dims: Tuple[int, int, int],
        hidden_dim: int,
        dropout_rate: float,
        num_classes: int,
        pad_index: Tensor,
        encoder: EfficientNet,
        decoder: Decoder,
    ) -> None:
        # TODO: Load pretrained vqvae encoder.
        super().__init__(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            pad_index=pad_index,
            encoder=encoder,
            decoder=decoder,
        )
        # Latent projector for down sampling number of filters and 2d
        # positional encoding.
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.out_channels,
                out_channels=self.hidden_dim,
                kernel_size=1,
            ),
            PositionalEncoding2D(
                hidden_dim=self.hidden_dim,
                max_h=self.input_dims[1],
                max_w=self.input_dims[2],
            ),
            nn.Flatten(start_dim=2),
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encodes an image into a latent feature vector.

        Args:
            x (Tensor): Image tensor.

        Shape:
            - x: :math: `(B, C, H, W)`
            - z: :math: `(B, Sx, E)`

            where Sx is the length of the flattened feature maps projected from
            the encoder. E latent dimension for each pixel in the projected
            feature maps.

        Returns:
            Tensor: A Latent embedding of the image.
        """
        z = self.encoder(x)
        z = self.latent_encoder(z)

        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z

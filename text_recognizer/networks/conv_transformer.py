"""Vision transformer for character recognition."""
import math
from typing import Optional, Tuple, Type

from loguru import logger as log
from torch import nn, Tensor

from text_recognizer.networks.base import BaseTransformer
from text_recognizer.networks.transformer.axial_attention.encoder import AxialEncoder
from text_recognizer.networks.transformer.decoder import Decoder
from text_recognizer.networks.transformer.embeddings.axial import (
    AxialPositionalEmbedding,
)


class ConvTransformer(BaseTransformer):
    """Convolutional encoder and transformer decoder network."""

    def __init__(
        self,
        input_dims: Tuple[int, int, int],
        hidden_dim: int,
        num_classes: int,
        pad_index: Tensor,
        encoder: Type[nn.Module],
        decoder: Decoder,
        axial_encoder: Optional[AxialEncoder],
        pixel_pos_embedding: AxialPositionalEmbedding,
        token_pos_embedding: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__(
            input_dims,
            hidden_dim,
            num_classes,
            pad_index,
            encoder,
            decoder,
            token_pos_embedding,
        )

        self.pixel_pos_embedding = pixel_pos_embedding
        self.axial_encoder = axial_encoder

        # Latent projector for down sampling number of filters and 2d
        # positional encoding.
        self.conv = nn.Conv2d(
            in_channels=self.encoder.out_channels,
            out_channels=self.hidden_dim,
            kernel_size=1,
        )

        # Initalize weights for encoder.
        self.init_weights()

    def init_weights(self) -> None:
        """Initalize weights for decoder network and to_logits."""
        bound = 0.1
        self.token_embedding.weight.data.uniform_(-bound, bound)
        self.to_logits.bias.data.zero_()
        self.to_logits.weight.data.uniform_(-bound, bound)

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
        z = self.conv(z)
        z = self.pixel_pos_embedding(z)
        z = self.axial_encoder(z) if self.axial_encoder is not None else z
        z = z.flatten(start_dim=2)

        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z

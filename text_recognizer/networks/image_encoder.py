"""Encodes images to latent embeddings."""
from typing import Tuple, Type

from torch import Tensor, nn

from text_recognizer.networks.transformer.embeddings.axial import (
    AxialPositionalEmbeddingImage,
)


class ImageEncoder(nn.Module):
    """Base transformer network."""

    def __init__(
        self,
        encoder: Type[nn.Module],
        pixel_embedding: AxialPositionalEmbeddingImage,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pixel_embedding = pixel_embedding

    def forward(self, img: Tensor) -> Tensor:
        """Encodes an image into a latent feature vector.

        Args:
            img (Tensor): Image tensor.

        Shape:
            - x: :math: `(B, C, H, W)`
            - z: :math: `(B, Sx, D)`

            where Sx is the length of the flattened feature maps projected from
            the encoder. D latent dimension for each pixel in the projected
            feature maps.

        Returns:
            Tensor: A Latent embedding of the image.
        """
        z = self.encoder(img)
        z = z + self.pixel_embedding(z)
        z = z.flatten(start_dim=2)
        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z

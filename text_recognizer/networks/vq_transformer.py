"""Vector quantized encoder, transformer decoder."""
from typing import Tuple

import torch
from torch import Tensor

from text_recognizer.networks.vqvae.vqvae import VQVAE
from text_recognizer.networks.conv_transformer import ConvTransformer
from text_recognizer.networks.transformer.layers import Decoder


class VqTransformer(ConvTransformer):
    """Convolutional encoder and transformer decoder network."""

    def __init__(
        self,
        input_dims: Tuple[int, int, int],
        encoder_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        num_classes: int,
        pad_index: Tensor,
        encoder: VQVAE,
        decoder: Decoder,
        pretrained_encoder_path: str,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            pad_index=pad_index,
            encoder=encoder,
            decoder=decoder,
        )
        self.pretrained_encoder_path = pretrained_encoder_path

        # For typing
        self.encoder: VQVAE

    def setup_encoder(self) -> None:
        """Remove unecessary layers."""
        # TODO: load pretrained vqvae
        del self.encoder.decoder
        del self.encoder.post_codebook_conv

    def encode(self, x: Tensor) -> Tensor:
        """Encodes an image into a discrete (VQ) latent representation.

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
        with torch.no_grad():
            z_e = self.encoder.encode(x)
            z_q, _ = self.encoder.quantize(z_e)

        z = self.latent_encoder(z_q)

        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z

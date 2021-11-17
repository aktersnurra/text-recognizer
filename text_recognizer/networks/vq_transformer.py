"""Vector quantized encoder, transformer decoder."""
from typing import Optional, Tuple, Type

from torch import nn, Tensor

from text_recognizer.networks.conv_transformer import ConvTransformer
from text_recognizer.networks.quantizer.quantizer import VectorQuantizer
from text_recognizer.networks.transformer.layers import Decoder


class VqTransformer(ConvTransformer):
    """Convolutional encoder and transformer decoder network."""

    def __init__(
        self,
        input_dims: Tuple[int, int, int],
        hidden_dim: int,
        num_classes: int,
        pad_index: Tensor,
        encoder: nn.Module,
        decoder: Decoder,
        pixel_pos_embedding: Type[nn.Module],
        quantizer: VectorQuantizer,
        token_pos_embedding: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pad_index=pad_index,
            encoder=encoder,
            decoder=decoder,
            pixel_pos_embedding=pixel_pos_embedding,
            token_pos_embedding=token_pos_embedding,
        )
        self.quantizer = quantizer

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
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
        z = self.encoder(x)
        z = self.conv(z)
        z, _, commitment_loss = self.quantizer(z)
        z = self.pixel_pos_embedding(z)
        z = z.flatten(start_dim=2)

        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z, commitment_loss

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Encodes images into word piece logtis.

        Args:
            x (Tensor): Input image(s).
            context (Tensor): Target word embeddings.

        Shapes:
            - x: :math: `(B, C, H, W)`
            - context: :math: `(B, Sy, T)`

            where B is the batch size, C is the number of input channels, H is
            the image height and W is the image width.

        Returns:
            Tensor: Sequence of logits.
        """
        z, commitment_loss = self.encode(x)
        logits = self.decode(z, context)
        return logits, commitment_loss

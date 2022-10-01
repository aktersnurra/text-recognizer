"""Base network module."""
from typing import Type

from torch import Tensor, nn

from text_recognizer.networks.transformer.decoder import Decoder


class ConvTransformer(nn.Module):
    """Base transformer network."""

    def __init__(
        self,
        encoder: Type[nn.Module],
        decoder: Decoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, img: Tensor) -> Tensor:
        """Encodes images to latent representation."""
        return self.encoder(img)

    def decode(self, tokens: Tensor, img_features: Tensor) -> Tensor:
        """Decodes latent images embedding into characters."""
        return self.decoder(tokens, img_features)

    def forward(self, img: Tensor, tokens: Tensor) -> Tensor:
        """Encodes images into token logtis.

        Args:
            img (Tensor): Input image(s).
            tokens (Tensor): token embeddings.

        Shapes:
            - img: :math: `(B, 1, H, W)`
            - tokens: :math: `(B, Sy)`
            - logits: :math: `(B, Sy, C)`

            where B is the batch size, H is the image height, W is the image
            width, Sy the output length, and C is the number of classes.

        Returns:
            Tensor: Sequence of logits.
        """
        img_features = self.encode(img)
        logits = self.decode(tokens, img_features)
        return logits

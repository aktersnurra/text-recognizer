"""Vision transformer for character recognition."""
import math
from typing import Optional, Tuple, Type

from loguru import logger as log
from torch import nn, Tensor

from text_recognizer.networks.transformer.layers import Decoder


class ConvTransformer(nn.Module):
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
        token_pos_embedding: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pad_index = pad_index
        self.encoder = encoder
        self.decoder = decoder
        self.pixel_pos_embedding = pixel_pos_embedding

        # Latent projector for down sampling number of filters and 2d
        # positional encoding.
        self.conv = nn.Conv2d(
            in_channels=self.encoder.out_channels,
            out_channels=self.hidden_dim,
            kernel_size=1,
        )

        # Token embedding.
        self.token_embedding = nn.Embedding(
            num_embeddings=self.num_classes, embedding_dim=self.hidden_dim
        )

        # Positional encoding for decoder tokens.
        if not self.decoder.has_pos_emb:
            self.token_pos_embedding = token_pos_embedding
        else:
            self.token_pos_embedding = None
            log.debug("Decoder already have a positional embedding.")

        # Head
        self.head = nn.Linear(
            in_features=self.hidden_dim, out_features=self.num_classes
        )

        # Initalize weights for encoder.
        self.init_weights()

    def init_weights(self) -> None:
        """Initalize weights for decoder network and head."""
        bound = 0.1
        self.token_embedding.weight.data.uniform_(-bound, bound)
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-bound, bound)

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
        z = z.flatten(start_dim=2)

        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z

    def decode(self, src: Tensor, trg: Tensor) -> Tensor:
        """Decodes latent images embedding into word pieces.

        Args:
            src (Tensor): Latent images embedding.
            trg (Tensor): Word embeddings.

        Shapes:
            - z: :math: `(B, Sx, E)`
            - context: :math: `(B, Sy)`
            - out: :math: `(B, Sy, T)`

            where Sy is the length of the output and T is the number of tokens.

        Returns:
            Tensor: Sequence of word piece embeddings.
        """
        trg = trg.long()
        trg_mask = trg != self.pad_index
        trg = self.token_embedding(trg) * math.sqrt(self.hidden_dim)
        trg = (
            self.token_pos_embedding(trg)
            if self.token_pos_embedding is not None
            else trg
        )
        out = self.decoder(x=trg, context=src, input_mask=trg_mask)
        logits = self.head(out)  # [B, Sy, T]
        logits = logits.permute(0, 2, 1)  # [B, T, Sy]
        return logits

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
        z = self.encode(x)
        logits = self.decode(z, context)
        return logits

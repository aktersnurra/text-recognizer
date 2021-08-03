"""Vision transformer for character recognition."""
import math
from typing import Tuple

from torch import nn, Tensor

from text_recognizer.networks.encoders.efficientnet import EfficientNet
from text_recognizer.networks.transformer.layers import Decoder
from text_recognizer.networks.transformer.positional_encodings import (
    PositionalEncoding,
    PositionalEncoding2D,
)


class ConvTransformer(nn.Module):
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
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.pad_index = pad_index
        self.encoder = encoder
        self.decoder = decoder

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

        # Token embedding.
        self.token_embedding = nn.Embedding(
            num_embeddings=self.num_classes, embedding_dim=self.hidden_dim
        )

        # Positional encoding for decoder tokens.
        self.token_pos_encoder = PositionalEncoding(
            hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate
        )
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
        # TODO: Initalize encoder?

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

    def decode(self, z: Tensor, context: Tensor) -> Tensor:
        """Decodes latent images embedding into word pieces.

        Args:
            z (Tensor): Latent images embedding.
            context (Tensor): Word embeddings.

        Shapes:
            - z: :math: `(B, Sx, E)`
            - context: :math: `(B, Sy)`
            - out: :math: `(B, Sy, T)`

            where Sy is the length of the output and T is the number of tokens.

        Returns:
            Tensor: Sequence of word piece embeddings.
        """
        context = context.long()
        context_mask = context != self.pad_index
        context = self.token_embedding(context) * math.sqrt(self.hidden_dim)
        context = self.token_pos_encoder(context)
        out = self.decoder(x=context, context=z, mask=context_mask)
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

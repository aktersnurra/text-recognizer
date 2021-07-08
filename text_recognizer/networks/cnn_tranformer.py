"""Vision transformer for character recognition."""
import math
from typing import Tuple, Type

import attr
from torch import nn, Tensor

from text_recognizer.data.mappings import AbstractMapping
from text_recognizer.networks.transformer.layers import Decoder
from text_recognizer.networks.transformer.positional_encodings import (
    PositionalEncoding,
    PositionalEncoding2D,
)


@attr.s
class CnnTransformer(nn.Module):
    def __attrs_pre_init__(self) -> None:
        super().__init__()

    # Parameters,
    input_dims: Tuple[int, int, int] = attr.ib()
    hidden_dim: int = attr.ib()
    dropout_rate: float = attr.ib()
    max_output_len: int = attr.ib()
    num_classes: int = attr.ib()
    padding_idx: int = attr.ib()

    # Modules.
    encoder: Type[nn.Module] = attr.ib()
    decoder: Decoder = attr.ib()
    embedding: nn.Embedding = attr.ib(init=False, default=None)
    latent_encoder: nn.Sequential = attr.ib(init=False, default=None)
    token_embedding: nn.Embedding = attr.ib(init=False, default=None)
    token_pos_encoder: PositionalEncoding = attr.ib(init=False, default=None)
    head: nn.Linear = attr.ib(init=False, default=None)
    mapping: AbstractMapping = attr.ib(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
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

        # Permute tensor from [B, E, Ho * Wo] to [Sx, B, E]
        z = z.permute(2, 0, 1)
        return z

    def decode(self, z: Tensor, trg: Tensor) -> Tensor:
        """Decodes latent images embedding into word pieces.

        Args:
            z (Tensor): Latent images embedding.
            trg (Tensor): Word embeddings.

        Shapes:
            - z: :math: `(B, Sx, E)`
            - trg: :math: `(B, Sy)`
            - out: :math: `(B, Sy, T)`

            where Sy is the length of the output and T is the number of tokens.

        Returns:
            Tensor: Sequence of word piece embeddings.
        """
        trg_mask = trg != self.padding_idx
        trg = self.token_embedding(trg) * math.sqrt(self.hidden_dim)
        trg = self.token_pos_encoder(trg)
        out = self.decoder(x=trg, context=z, mask=trg_mask)
        logits = self.head(out)
        return logits

    def forward(self, x: Tensor, trg: Tensor) -> Tensor:
        """Encodes images into word piece logtis.

        Args:
            x (Tensor): Input image(s).
            trg (Tensor): Target word embeddings.

        Shapes:
            - x: :math: `(B, C, H, W)`
            - trg: :math: `(B, Sy, T)`

            where B is the batch size, C is the number of input channels, H is
            the image height and W is the image width.
        """
        z = self.encode(x)
        logits = self.decode(z, trg)
        return logits

    def predict(self, x: Tensor) -> Tensor:
        """Predicts text in image."""
        # TODO: continue here!!!!!!!!!
        pass

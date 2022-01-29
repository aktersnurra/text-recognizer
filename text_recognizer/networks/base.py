"""Base network module."""
import math
from typing import Optional, Tuple, Type

from loguru import logger as log
from torch import nn, Tensor

from text_recognizer.networks.transformer.decoder import Decoder


class BaseTransformer(nn.Module):
    """Base transformer network."""

    def __init__(
        self,
        input_dims: Tuple[int, int, int],
        hidden_dim: int,
        num_classes: int,
        pad_index: Tensor,
        encoder: Type[nn.Module],
        decoder: Decoder,
        token_pos_embedding: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pad_index = pad_index
        self.encoder = encoder
        self.decoder = decoder

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

        # Output layer
        self.to_logits = nn.Linear(
            in_features=self.hidden_dim, out_features=self.num_classes
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encodes images with encoder."""
        return self.encoder(x)

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
        logits = self.to_logits(out)  # [B, Sy, T]
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

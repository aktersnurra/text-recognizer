"""Text decoder."""
from typing import Optional, Type

import torch
from torch import Tensor, nn

from text_recognizer.networks.transformer.decoder import Decoder


class TextDecoder(nn.Module):
    """Decoder transformer network."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        pad_index: Tensor,
        decoder: Decoder,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pad_index = pad_index
        self.decoder = decoder
        self.token_embedding = nn.Embedding(
            num_embeddings=self.num_classes, embedding_dim=self.hidden_dim
        )
        self.to_logits = nn.Linear(
            in_features=self.hidden_dim, out_features=self.num_classes
        )

    def forward(self, tokens: Tensor, img_features: Tensor) -> Tensor:
        """Decodes latent images embedding into word pieces.

        Args:
            tokens (Tensor): Token indecies.
            img_features (Tensor): Latent images embedding.

        Shapes:
            - tokens: :math: `(B, Sy)`
            - img_features: :math: `(B, Sx, D)`
            - logits: :math: `(B, Sy, C)`

            where Sy is the length of the output, C is the number of classes
            and D is the hidden dimension.

        Returns:
            Tensor: Sequence of logits.
        """
        tokens = tokens.long()
        mask = tokens != self.pad_index
        tokens = self.token_embedding(tokens)
        tokens = self.decoder(x=tokens, context=img_features, mask=mask)
        logits = (
            tokens @ torch.transpose(self.token_embedding.weight.to(tokens.dtype), 0, 1)
        ).float()
        logits = self.to_logits(tokens)  # [B, Sy, C]
        logits = logits.permute(0, 2, 1)  # [B, C, Sy]
        return logits

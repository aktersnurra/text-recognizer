"""Perceiver network module."""
from typing import Optional, Tuple, Type

from torch import nn, Tensor

from text_recognizer.networks.perceiver.perceiver import PerceiverIO
from text_recognizer.networks.transformer.embeddings.axial import (
    AxialPositionalEmbedding,
)


class ConvPerceiver(nn.Module):
    """Base transformer network."""

    def __init__(
        self,
        input_dims: Tuple[int, int, int],
        hidden_dim: int,
        queries_dim: int,
        num_classes: int,
        pad_index: Tensor,
        encoder: Type[nn.Module],
        decoder: PerceiverIO,
        max_length: int,
        pixel_embedding: AxialPositionalEmbedding,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pad_index = pad_index
        self.max_length = max_length
        self.encoder = encoder
        self.decoder = decoder
        self.pixel_embedding = pixel_embedding
        self.to_queries = nn.Linear(self.hidden_dim, queries_dim)
        self.conv = nn.Conv2d(
            in_channels=self.encoder.out_channels,
            out_channels=self.hidden_dim,
            kernel_size=1,
        )

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.conv(z)
        z = self.pixel_embedding(z)
        z = z.flatten(start_dim=2)

        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z

    def decode(self, z: Tensor) -> Tensor:
        queries = self.to_queries(z[:, : self.max_length, :])
        logits = self.decoder(data=z, queries=queries)
        logits = logits.permute(0, 2, 1)  # [B, C, Sy]
        return logits

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        logits = self.decode(z)
        return logits

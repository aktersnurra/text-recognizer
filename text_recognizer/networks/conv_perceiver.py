"""Perceiver network module."""
from typing import Optional, Tuple, Type

from einops import repeat
import torch
from torch import nn, Tensor

from text_recognizer.networks.perceiver.perceiver import PerceiverIO
from text_recognizer.networks.transformer.embeddings.absolute import (
    AbsolutePositionalEmbedding,
)
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
        num_queries: int,
        num_classes: int,
        pad_index: Tensor,
        encoder: Type[nn.Module],
        decoder: PerceiverIO,
        max_length: int,
        pixel_embedding: AxialPositionalEmbedding,
        query_pos_emb: AbsolutePositionalEmbedding,
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
        self.query_pos_emb = query_pos_emb
        self.queries = nn.Parameter(torch.randn(num_queries, queries_dim))

    def encode(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = torch.concat([z, self.pixel_embedding(z)], dim=1)
        z = z.flatten(start_dim=2)
        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z

    def decode(self, z: Tensor) -> Tensor:
        b = z.shape[0]
        queries = repeat(self.queries, "n d -> b n d", b=b)
        pos_emb = repeat(self.query_pos_emb(queries), "n d -> b n d", b=b)
        queries = torch.concat([queries, pos_emb], dim=-1)
        logits = self.decoder(data=z, queries=queries)
        logits = logits.permute(0, 2, 1)  # [B, C, Sy]
        return logits

    def forward(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        logits = self.decode(z)
        return logits

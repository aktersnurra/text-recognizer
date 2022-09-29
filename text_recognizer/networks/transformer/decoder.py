"""Transformer decoder module."""
from copy import deepcopy
from typing import Optional, Type

from torch import Tensor, nn

from text_recognizer.networks.transformer.attention import Attention
from text_recognizer.networks.transformer.decoder_block import DecoderBlock
from text_recognizer.networks.transformer.embeddings.rotary import RotaryEmbedding
from text_recognizer.networks.transformer.ff import FeedForward


class Decoder(nn.Module):
    """Decoder Network."""

    def __init__(
        self,
        depth: int,
        dim: int,
        block: DecoderBlock,
        rotary_embedding: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.rotary_embedding = rotary_embedding
        self.blocks = nn.ModuleList([deepcopy(block) for _ in range(self.depth)])
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies the network to the signals."""
        for block in self.blocks:
            x = block(
                x=x,
                context=context,
                mask=mask,
                rotary_embedding=self.rotary_embedding,
            )
        return self.ln(x)

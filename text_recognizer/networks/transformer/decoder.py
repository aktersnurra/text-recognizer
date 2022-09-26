"""Transformer decoder module."""
from copy import deepcopy
from typing import Optional, Type

from torch import Tensor, nn

from text_recognizer.networks.transformer.attention import Attention
from text_recognizer.networks.transformer.decoder_block import DecoderBlock
from text_recognizer.networks.transformer.ff import FeedForward


class Decoder(nn.Module):
    """Decoder Network."""

    def __init__(self, depth: int, dim: int, block: DecoderBlock) -> None:
        super().__init__()
        self.depth = depth
        self.has_pos_emb = block.has_pos_emb
        self.blocks = nn.ModuleList([deepcopy(block) for _ in range(self.depth)])
        self.ln = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies the network to the signals."""
        for block in self.blocks:
            x = block(
                x=x, context=context, input_mask=input_mask, context_mask=context_mask
            )
        return self.ln(x)

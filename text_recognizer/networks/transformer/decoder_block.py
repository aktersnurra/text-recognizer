"""Transformer decoder module."""
from copy import deepcopy
from typing import Optional, Type

from torch import Tensor, nn

from text_recognizer.networks.transformer.attention import Attention
from text_recognizer.networks.transformer.embeddings.rotary import RotaryEmbedding
from text_recognizer.networks.transformer.ff import FeedForward


class DecoderBlock(nn.Module):
    """Decoder block."""

    def __init__(
        self,
        self_attn: Attention,
        norm: Type[nn.Module],
        ff: FeedForward,
        cross_attn: Optional[Attention] = None,
    ) -> None:
        super().__init__()
        self.ln_attn = norm
        self.attn = self_attn
        self.ln_cross_attn = deepcopy(norm)
        self.cross_attn = cross_attn
        self.ln_ff = deepcopy(norm)
        self.ff = ff

    def forward(
        self,
        x: Tensor,
        rotary_embedding: RotaryEmbedding,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        x = x + self.attn(self.ln_attn(x), mask=mask, rotary_embedding=rotary_embedding)
        x = x + self.cross_attn(
            x=self.ln_cross_attn(x),
            context=context,
        )
        x = x + self.ff(self.ln_ff(x))
        return x

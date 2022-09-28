"""Transformer decoder module."""
from copy import deepcopy
from typing import Optional, Type

from torch import Tensor, nn

from text_recognizer.networks.transformer.attention import Attention
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
        self.has_pos_emb = self.attn.rotary_embedding is not None

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        x = x + self.attn(self.ln_attn(x), input_mask=input_mask)
        x = x + self.cross_attn(
            x=self.ln_cross_attn(x),
            context=context,
            input_mask=input_mask,
            context_mask=context_mask,
        )
        x = x + self.ff(self.ln_ff(x))
        return x

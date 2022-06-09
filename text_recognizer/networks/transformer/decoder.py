"""Transformer decoder module."""
from copy import deepcopy
from typing import Optional, Type

from torch import nn, Tensor

from text_recognizer.networks.transformer.attention import Attention
from text_recognizer.networks.transformer.mlp import FeedForward


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
        self.layers = ("self_attn", "cross_attn", "ff")
        self.has_pos_emb = self_attn.rotary_embedding is not None
        self.blocks = self._build(self_attn, norm, ff, cross_attn)

    def _build(
        self,
        self_attn: Attention,
        norm: Type[nn.Module],
        ff: FeedForward,
        cross_attn: Optional[Attention],
    ) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                self.layers[0]: nn.ModuleList([norm, self_attn]),
                self.layers[1]: nn.ModuleList([deepcopy(norm), cross_attn]),
                self.layers[2]: nn.ModuleList([deepcopy(norm), ff]),
            }
        )

    def _apply_block(
        self,
        layer: str,
        x: Tensor,
        context: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies block function."""
        residual = x
        norm_fn, layer_fn = self.blocks[layer]
        if layer == "self_attn":
            out = layer_fn(x=x, input_mask=input_mask)
        elif layer == "cross_attn":
            out = layer_fn(
                x=x, context=context, input_mask=input_mask, context_mask=context_mask
            )
        else:
            out = layer_fn(x)
        out += residual
        return norm_fn(out)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        for layer in self.layers:
            x = self._apply_block(
                layer=layer,
                x=x,
                context=context,
                input_mask=input_mask,
                context_mask=context_mask,
            )
        return x


class Decoder(nn.Module):
    """Decoder Network."""

    def __init__(self, depth: int, block: DecoderBlock) -> None:
        super().__init__()
        self.depth = depth
        self.has_pos_emb = block.has_pos_emb
        self.blocks = nn.ModuleList([deepcopy(block) for _ in range(self.depth)])

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
        return x

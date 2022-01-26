"""Transformer attention layer."""
from copy import deepcopy
from typing import Any, Optional, Tuple, Type

from torch import nn, Tensor

from text_recognizer.networks.transformer.attention import Attention
from text_recognizer.networks.transformer.mlp import FeedForward
from text_recognizer.networks.transformer.residual import Residual


class AttentionLayers(nn.Module):
    """Standard transfomer layer."""

    def __init__(
        self,
        depth: int,
        self_attn: Attention,
        norm: Type[nn.Module],
        ff: FeedForward,
        cross_attn: Optional[Attention] = None,
        pre_norm: bool = True,
        has_pos_emb: bool = True,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.has_pos_emb = has_pos_emb
        self.layer_types = self._get_layer_types() * depth
        self.layers = self._build(self_attn, norm, ff, cross_attn)

    def _get_layer_types(self) -> Tuple:
        """Get layer specification."""
        if self.cross_attn is not None:
            return "a", "c", "f"
        return "a", "f"

    def _build(
        self,
        self_attn: Attention,
        norm: Type[nn.Module],
        ff: FeedForward,
        cross_attn: Optional[Attention],
    ) -> nn.ModuleList:
        """Configures transformer network."""
        layers = nn.ModuleList([])
        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = deepcopy(self_attn)
            elif layer_type == "c":
                layer = deepcopy(cross_attn)
            elif layer_type == "f":
                layer = deepcopy(ff)
            layers.append(nn.ModuleList([deepcopy(norm), layer, Residual()]))
        return layers

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass."""
        for i, (layer_type, (norm, block, residual_fn)) in enumerate(
            zip(self.layer_types, self.layers)
        ):
            is_last = i == len(self.layers) - 1
            residual = x

            if self.pre_norm:
                x = norm(x)

            if layer_type == "a":
                out = block(x=x, input_mask=input_mask)
            elif layer_type == "c":
                out = block(
                    x, context=context, input_mask=input_mask, context_mask=context_mask
                )
            elif layer_type == "f":
                out = block(x)

            x = residual_fn(out, residual)

            if not self.pre_norm and not is_last:
                x = norm(x)

        return x


class Decoder(AttentionLayers):
    """Decoder module."""

    def __init__(self, **kwargs: Any) -> None:
        if "cross_attn" not in kwargs:
            ValueError("Decoder requires cross attention.")

        super().__init__(**kwargs)


class Encoder(AttentionLayers):
    """Encoder module."""

    def __init__(self, **kwargs: Any) -> None:
        if "cross_attn" in kwargs:
            ValueError("Encoder requires cross attention.")

        super().__init__(**kwargs)

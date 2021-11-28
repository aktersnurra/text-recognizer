"""Transformer attention layer."""
from copy import deepcopy
from typing import Any, Optional, Tuple, Type

import attr
from torch import nn, Tensor

from text_recognizer.networks.transformer.attention import Attention
from text_recognizer.networks.transformer.mlp import FeedForward
from text_recognizer.networks.transformer.residual import Residual


@attr.s(eq=False)
class AttentionLayers(nn.Module):
    """Standard transfomer layer."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    depth: int = attr.ib()
    self_attn: Attention = attr.ib()
    norm: Type[nn.Module] = attr.ib()
    ff: FeedForward = attr.ib()
    cross_attn: Optional[Attention] = attr.ib(default=None)
    pre_norm: bool = attr.ib(default=True)
    has_pos_emb: bool = attr.ib(default=False)
    layer_types: Tuple[str, ...] = attr.ib(init=False)
    layers: nn.ModuleList = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.layer_types = self._get_layer_types() * self.depth
        self.layers = self._build()

    def _get_layer_types(self) -> Tuple:
        """Get layer specification."""
        if self.cross_attn is not None:
            return "a", "c", "f"
        return "a", "f"

    def _delete(self) -> None:
        del self.self_attn
        del self.ff
        del self.norm
        del self.cross_attn

    def _build(self) -> nn.ModuleList:
        """Configures transformer network."""
        layers = nn.ModuleList([])
        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = deepcopy(self.self_attn)
            elif layer_type == "c":
                layer = deepcopy(self.cross_attn)
            elif layer_type == "f":
                layer = deepcopy(self.ff)
            layers.append(nn.ModuleList([deepcopy(self.norm), layer, Residual()]))
        self._delete()
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

"""Transformer attention layer."""
from copy import deepcopy
from typing import Any, Optional, Tuple, Type

import attr
from torch import nn, Tensor

from text_recognizer.networks.transformer.attention import Attention
from text_recognizer.networks.transformer.local_attention import LocalAttention
from text_recognizer.networks.transformer.mlp import FeedForward
from text_recognizer.networks.transformer.residual import Residual


@attr.s(eq=False)
class AttentionLayers(nn.Module):
    """Standard transfomer layer."""

    def __attrs_pre_init__(self) -> None:
        """Pre init constructor."""
        super().__init__()

    depth: int = attr.ib()
    self_attn: Attention = attr.ib()
    norm: Type[nn.Module] = attr.ib()
    ff: FeedForward = attr.ib()
    cross_attn: Optional[Attention] = attr.ib(default=None)
    local_self_attn: Optional[LocalAttention] = attr.ib(default=None)
    pre_norm: bool = attr.ib(default=True)
    local_depth: Optional[int] = attr.ib(default=None)
    has_pos_emb: bool = attr.ib(default=False)
    layer_types: Tuple[str, ...] = attr.ib(init=False)
    layers: nn.ModuleList = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        if self.local_self_attn is not None:
            if self.local_depth is None:
                ValueError("Local depth has to be specified")
        self.layer_types = self._get_layer_types() * self.depth
        self.layers = self._build_network()

    def _get_layer_types(self) -> Tuple:
        """Get layer specification."""
        if self.cross_attn is not None:
            return "a", "c", "f"
        return "a", "f"

    def _self_attn_block(self, i: int) -> Type[nn.Module]:
        if self.local_depth is not None and i < self.local_depth:
            return deepcopy(self.local_self_attn)
        return deepcopy(self.self_attn)

    def _delete(self) -> None:
        del self.self_attn
        del self.local_self_attn
        del self.ff
        del self.norm
        del self.cross_attn

    def _build_network(self) -> nn.ModuleList:
        """Configures transformer network."""
        layers = nn.ModuleList([])
        self_attn_depth = 0
        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = self._self_attn_block(self_attn_depth)
                self_attn_depth += 1
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
        super().__init__(**kwargs)

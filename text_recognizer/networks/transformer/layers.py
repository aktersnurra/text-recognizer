"""Transformer attention layer."""
from functools import partial
from typing import Any, Dict, Optional, Tuple

import attr
from torch import nn, Tensor

from text_recognizer.networks.transformer.residual import Residual
from text_recognizer.networks.transformer.positional_encodings.rotary_embedding import (
    RotaryEmbedding,
)
from text_recognizer.networks.util import load_partial_fn


@attr.s
class AttentionLayers(nn.Module):
    """Standard transfomer layer."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    dim: int = attr.ib()
    depth: int = attr.ib()
    num_heads: int = attr.ib()
    attn_fn: str = attr.ib()
    attn_kwargs: Dict = attr.ib()
    norm_fn: str = attr.ib()
    ff_fn: str = attr.ib()
    ff_kwargs: Dict = attr.ib()
    causal: bool = attr.ib(default=False)
    cross_attend: bool = attr.ib(default=False)
    pre_norm: bool = attr.ib(default=True)
    rotary_emb: Optional[RotaryEmbedding] = attr.ib(default=None, init=False)
    has_pos_emb: bool = attr.ib(init=False)
    layer_types: Tuple[str, ...] = attr.ib(init=False)
    layers: nn.ModuleList = attr.ib(init=False)
    attn: partial = attr.ib(init=False)
    norm: partial = attr.ib(init=False)
    ff: partial = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        self.has_pos_emb = True if self.rotary_emb is not None else False
        self.layer_types = self._get_layer_types() * self.depth
        attn = load_partial_fn(
            self.attn_fn, dim=self.dim, num_heads=self.num_heads, **self.attn_kwargs
        )
        norm = load_partial_fn(self.norm_fn, dim=self.dim)
        ff = load_partial_fn(self.ff_fn, dim=self.dim, **self.ff_kwargs)
        self.layers = self._build_network(attn, norm, ff)

    def _get_layer_types(self) -> Tuple:
        """Get layer specification."""
        if self.cross_attend:
            return "a", "c", "f"
        return "a", "f"

    def _build_network(
        self, attn: partial, norm: partial, ff: partial,
    ) -> nn.ModuleList:
        """Configures transformer network."""
        layers = nn.ModuleList([])
        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = attn(causal=self.causal)
            elif layer_type == "c":
                layer = attn()
            elif layer_type == "f":
                layer = ff()
            residual_fn = Residual()
            layers.append(nn.ModuleList([norm(), layer, residual_fn]))
        return layers

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        rotary_pos_emb = self.rotary_emb(x) if self.rotary_emb is not None else None
        for i, (layer_type, (norm, block, residual_fn)) in enumerate(
            zip(self.layer_types, self.layers)
        ):
            is_last = i == len(self.layers) - 1
            residual = x

            if self.pre_norm:
                x = norm(x)

            if layer_type == "a":
                out, _ = block(x=x, mask=mask, rotary_pos_emb=rotary_pos_emb)
            elif layer_type == "c":
                out, _ = block(x, context=context, mask=mask, context_mask=context_mask)
            elif layer_type == "f":
                out = block(x)

            x = residual_fn(out, residual)

            if not self.pre_norm and not is_last:
                x = norm(x)

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs: Any) -> None:
        assert "causal" not in kwargs, "Cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs: Any) -> None:
        assert "causal" not in kwargs, "Cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)

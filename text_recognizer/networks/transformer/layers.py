"""Generates the attention layer architecture."""
from functools import partial
from typing import Dict, Optional, Type

from click.types import Tuple

import torch
from torch import nn, Tensor

from .attention import Attention
from .mlp import FeedForward
from .residual import Residual


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        ff_kwargs: Dict,
        attn_kwargs: Dict,
        attn_fn: Type[nn.Module] = Attention,
        norm_fn: Type[nn.Module] = nn.LayerNorm,
        ff_fn: Type[nn.Module] = FeedForward,
        residual_fn: Type[nn.Module] = Residual,
        causal: bool = False,
        cross_attend: bool = False,
    ) -> None:
        super().__init__()
        attn_fn = partial(attn_fn, dim=dim, num_heads=num_heads, **attn_kwargs)
        norm_fn = partial(norm_fn, dim=dim)
        ff_fn = partial(ff_fn, dim=dim, **ff_kwargs)
        layer_types = self._get_layer_types(cross_attend) * depth
        self.layers = self._build_network(
            layer_types, causal, attn_fn, norm_fn, ff_fn, residual_fn
        )

    @staticmethod
    def _get_layer_types(cross_attend: bool) -> Tuple:
        """Get layer specification."""
        if cross_attend:
            return "a", "c", "f"
        return "a", "f"

    @staticmethod
    def _build_network(
        layer_types: Tuple,
        causal: bool,
        attn_fn: partial,
        norm_fn: partial,
        ff_fn: partial,
        residual_fn: Type[nn.Module],
    ) -> nn.ModuleList:
        """Configures transformer layers."""
        layers = nn.ModuleList([])
        for layer_type in layer_types:
            if layer_type == "a":
                layer = attn_fn(causal=causal)
            elif layer_type == "c":
                layer = attn_fn()
            elif layer_type == "f":
                layer = ff_fn()

            residual_fn = residual_fn()

            layers.append(nn.ModuleList([norm_fn(), layer, residual_fn]))
        return layers

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        pass

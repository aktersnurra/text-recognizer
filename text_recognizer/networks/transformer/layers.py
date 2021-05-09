"""Generates the attention layer architecture."""
from functools import partial
from typing import Any, Dict, Optional, Type

from click.types import Tuple

import torch
from torch import nn, Tensor

from .attention import Attention
from .mlp import FeedForward
from .residual import Residual
from .rotary_embedding import RotaryEmbedding


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
        rotary_emb: Optional[Type[nn.Module]] = None,
        rotary_emb_dim: Optional[int] = None,
        causal: bool = False,
        cross_attend: bool = False,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        attn_fn = partial(attn_fn, dim=dim, num_heads=num_heads, **attn_kwargs)
        norm_fn = partial(norm_fn, dim=dim)
        ff_fn = partial(ff_fn, dim=dim, **ff_kwargs)
        self.layer_types = self._get_layer_types(cross_attend) * depth
        self.layers = self._build_network(causal, attn_fn, norm_fn, ff_fn, residual_fn)
        rotary_emb_dim = max(rotary_emb_dim, 32) if rotary_emb_dim is not None else None
        self.rotary_emb = RotaryEmbedding(rotary_emb_dim) if rotary_emb else None
        self.pre_norm = pre_norm
        self.has_pos_emb = True if self.rotary_emb is not None else False

    @staticmethod
    def _get_layer_types(cross_attend: bool) -> Tuple:
        """Get layer specification."""
        if cross_attend:
            return "a", "c", "f"
        return "a", "f"

    def _build_network(
        self,
        causal: bool,
        attn_fn: partial,
        norm_fn: partial,
        ff_fn: partial,
        residual_fn: Type[nn.Module],
    ) -> nn.ModuleList:
        """Configures transformer network."""
        layers = nn.ModuleList([])
        for layer_type in self.layer_types:
            if layer_type == "a":
                layer = attn_fn(causal=causal)
            elif layer_type == "c":
                layer = attn_fn()
            elif layer_type == "f":
                layer = ff_fn()

            residual_fn = residual_fn()

            layers.append(nn.modulelist([norm_fn(), layer, residual_fn]))
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

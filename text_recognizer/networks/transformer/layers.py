"""Transformer attention layer."""
from typing import Any, Optional, Tuple, Type

import attr
from omegaconf.dictconfig import DictConfig
from torch import nn, Tensor

from text_recognizer.networks.transformer.embeddings.rotary import RotaryEmbedding
from text_recognizer.networks.transformer.residual import Residual
from text_recognizer.networks.util import load_partial_fn


@attr.s(eq=False)
class AttentionLayers(nn.Module):
    """Standard transfomer layer."""

    def __attrs_pre_init__(self) -> None:
        """Pre init constructor."""
        super().__init__()

    dim: int = attr.ib()
    depth: int = attr.ib()
    num_heads: int = attr.ib()
    attn_fn: str = attr.ib()
    attn_kwargs: DictConfig = attr.ib()
    norm_fn: str = attr.ib()
    ff_fn: str = attr.ib()
    ff_kwargs: DictConfig = attr.ib()
    rotary_emb: Optional[RotaryEmbedding] = attr.ib()
    local_attn_fn: Optional[str] = attr.ib(default=None)
    local_attn_kwargs: Optional[DictConfig] = attr.ib(default=None)
    causal: bool = attr.ib(default=False)
    cross_attend: bool = attr.ib(default=False)
    pre_norm: bool = attr.ib(default=True)
    local_depth: Optional[int] = attr.ib(default=None)
    layer_types: Tuple[str, ...] = attr.ib(init=False)
    layers: nn.ModuleList = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        self.layer_types = self._get_layer_types() * self.depth
        self.layers = self._build_network()

        if self.local_attn_kwargs is not None and self.local_attn_fn is not None:
            if "depth" not in self.local_attn_kwargs:
                ValueError("Local depth has to be specified")
            self.local_depth = self.local_attn_kwargs.pop("depth")

    def _get_layer_types(self) -> Tuple:
        """Get layer specification."""
        if self.cross_attend:
            return "a", "c", "f"
        return "a", "f"

    def _configure_causal_attn(self, i: int) -> Type[nn.Module]:
        if self.local_depth is not None and i <= self.local_depth:
            return load_partial_fn(
                self.local_attn_fn,
                dim=self.dim,
                num_heads=self.num_heads,
                **self.local_attn_kwargs,
            )()
        return load_partial_fn(
            self.attn_fn,
            causal=self.causal,
            dim=self.dim,
            num_heads=self.num_heads,
            **self.attn_kwargs,
        )()

    def _build_network(self) -> nn.ModuleList:
        """Configures transformer network."""
        norm = load_partial_fn(self.norm_fn, normalized_shape=self.dim)
        ff = load_partial_fn(self.ff_fn, dim=self.dim, **self.ff_kwargs)
        layers = nn.ModuleList([])
        for i, layer_type in enumerate(self.layer_types):
            if layer_type == "a":
                layer = self._configure_causal_attn(i)
            elif layer_type == "c":
                layer = load_partial_fn(
                    self.attn_fn,
                    dim=self.dim,
                    num_heads=self.num_heads,
                    **self.attn_kwargs,
                )()
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
        """Forward pass."""
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


@attr.s(auto_attribs=True, eq=False)
class Encoder(AttentionLayers):
    causal: bool = attr.ib(default=False, init=False)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs: Any) -> None:
        assert "causal" not in kwargs, "Cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)

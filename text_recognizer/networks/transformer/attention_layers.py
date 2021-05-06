"""Generates the attention layer architecture."""
from typing import Type

import torch
from torch import nn, Tensor


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        norm_layer: Type[nn.Module],
        causal: bool = False,
        cross_attend: bool = False,
        only_cross: bool = False,
    ) -> None:
        pass

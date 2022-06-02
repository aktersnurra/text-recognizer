"""Conformer block."""
from copy import deepcopy
from typing import Optional

from torch import nn, Tensor
from text_recognizer.networks.conformer.conv import ConformerConv

from text_recognizer.networks.conformer.mlp import MLP
from text_recognizer.networks.conformer.scale import Scale
from text_recognizer.networks.transformer.attention import Attention
from text_recognizer.networks.transformer.norm import PreNorm


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ff: MLP,
        attn: Attention,
        conv: ConformerConv,
    ) -> None:
        super().__init__()
        self.attn = PreNorm(dim, attn)
        self.ff_1 = Scale(0.5, ff)
        self.ff_2 = deepcopy(self.ff_1)
        self.conv = conv
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        x = self.ff_1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff_2(x) + x
        return self.post_norm(x)

from typing import Optional

import torch
from torch import nn, Tensor

from .l2_norm import l2_norm


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_length: int, use_l2: bool = False) -> None:
        super().__init__()
        self.scale = dim**-0.5 if not use_l2 else 1.0
        self.max_length = max_length
        self.use_l2 = use_l2
        self.to_embedding = nn.Embedding(max_length, dim)
        if self.use_l2:
            nn.init.normal_(self.to_embedding.weight, std=1e-5)

    def forward(self, x: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        n, device = x.shape[1], x.device
        assert (
            n <= self.max_length
        ), f"Sequence length {n} is greater than the maximum positional embedding {self.max_length}"

        if pos is None:
            pos = torch.arange(n, device=device)

        embedding = self.to_embedding(pos) * self.scale
        return l2_norm(embedding) if self.use_l2 else embedding

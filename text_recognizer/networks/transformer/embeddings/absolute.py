"""Absolute positional embedding."""
import torch
from torch import nn, Tensor


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self._weight_init()

    def _weight_init(self) -> None:
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]

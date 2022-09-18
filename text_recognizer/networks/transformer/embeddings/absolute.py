"""Absolute positional embedding."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed=False):
        super().__init__()
        self.scale = dim**-0.5 if not l2norm_embed else 1.0
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None):
        seq_len = x.shape[1]
        assert (
            seq_len <= self.max_seq_len
        ), f"you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"

        if pos is None:
            pos = torch.arange(seq_len, device=x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

"""Perceiver IO.

A copy from lucidrains.
"""
from itertools import repeat
from typing import Optional

import torch
from torch import nn, Tensor

from text_recognizer.networks.perceiver.attention import Attention
from text_recognizer.networks.transformer.ff import FeedForward
from text_recognizer.networks.transformer.norm import PreNorm


class PerceiverIO(nn.Module):
    def __init__(
        self,
        dim: int,
        cross_heads: int,
        cross_head_dim: int,
        num_latents: int,
        latent_dim: int,
        latent_heads: int,
        depth: int,
        queries_dim: int,
        logits_dim: int,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn_block = nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim, dim, heads=cross_heads, dim_head=cross_head_dim
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )

        self.layers = nn.ModuleList(
            [
                [
                    PreNorm(
                        latent_dim,
                        Attention(latent_dim, heads=latent_heads, dim_head=latent_dim),
                    ),
                    PreNorm(latent_dim, FeedForward(latent_dim)),
                ]
                for _ in range(depth)
            ]
        )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(
                queries_dim, latent_dim, heads=cross_heads, dim_head=cross_head_dim
            ),
            context_dim=latent_dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim))
        self.to_logits = nn.Linear(queries_dim, logits_dim)

    def forward(
        self, data: Tensor, queries: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        b = data.shape[0]
        x = repeat(self.latents, "nd -> bnd", b=b)

        cross_attn, cross_ff = self.cross_attn_block

        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        if queries.ndim == 2:
            queries = repeat(queries, "nd->bnd", b=b)

        latents = self.decoder_cross_attn(queries, context=x)
        latents = latents + self.decoder_ff(latents)

        return self.to_logits(latents)

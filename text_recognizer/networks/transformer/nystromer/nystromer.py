"""Nyströmer encoder.

Stolen from:
    https://github.com/lucidrains/nystrom-attention/blob/main/nystrom_attention/nystrom_attention.py

"""
from typing import Optional

from torch import nn, Tensor

from text_recognizer.networks.transformer.mlp import FeedForward
from text_recognizer.networks.transformer.norm import PreNorm
from text_recognizer.networks.transformer.nystromer.attention import NystromAttention


class Nystromer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        num_heads: int = 8,
        num_landmarks: int = 256,
        inverse_iter: int = 6,
        residual: bool = True,
        residual_conv_kernel: int = 33,
        dropout_rate: float = 0.0,
        glu: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            NystromAttention(
                                dim=dim,
                                dim_head=dim_head,
                                num_heads=num_heads,
                                num_landmarks=num_landmarks,
                                inverse_iter=inverse_iter,
                                residual=residual,
                                residual_conv_kernel=residual_conv_kernel,
                                dropout_rate=dropout_rate,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim=dim, glu=glu, dropout_rate=dropout_rate),
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x

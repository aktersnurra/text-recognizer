"""ConvNext module."""
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from .downsample import Downsample
from .norm import LayerNorm
from .transformer import Transformer


class GRN(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        spatial_l2_norm = x.norm(p=2, dim=(2, 3), keepdim=True)
        feat_norm = spatial_l2_norm / spatial_l2_norm.mean(dim=-1, keepdim=True).clamp(
            min=self.eps
        )
        return x * feat_norm * self.gamma + self.bias + x


class ConvNextBlock(nn.Module):
    """ConvNext block."""

    def __init__(self, dim: int, dim_out: int, mult: int) -> None:
        super().__init__()
        inner_dim = mult * dim_out
        self.ds_conv = nn.Conv2d(dim, dim, kernel_size=7, padding="same", groups=dim)
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            GRN(inner_dim),
            nn.Conv2d(inner_dim, dim_out, kernel_size=3, stride=1, padding="same"),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = self.ds_conv(x)
        h = self.net(h)
        return h + self.res_conv(x)


class ConvNext(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        dim_mults: Sequence[int] = (2, 4, 8),
        depths: Sequence[int] = (3, 3, 6),
        attn: Optional[Transformer] = None,
    ) -> None:
        super().__init__()
        dims = (dim, *map(lambda m: m * dim, dim_mults))
        self.attn = attn if attn is not None else nn.Identity()
        self.out_channels = dims[-1]
        self.stem = nn.Conv2d(1, dims[0], kernel_size=7, padding="same")
        self.layers = nn.ModuleList([])

        for i in range(len(dims) - 1):
            dim_in, dim_out = dims[i], dims[i + 1]
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [ConvNextBlock(dim_in, dim_in, 2) for _ in range(depths[i])]
                        ),
                        Downsample(dim_in, dim_out),
                    ]
                )
            )
        self.norm = LayerNorm(dims[-1])

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for blocks, down in self.layers:
            for fn in blocks:
                x = fn(x)
            x = down(x)
        x = self.attn(x)
        return self.norm(x)

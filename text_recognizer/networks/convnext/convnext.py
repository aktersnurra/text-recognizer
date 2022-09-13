from typing import Optional, Sequence

from text_recognizer.networks.convnext.attention import TransformerBlock
from torch import nn, Tensor

from text_recognizer.networks.convnext.downsample import Downsample
from text_recognizer.networks.convnext.norm import LayerNorm


class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, mult):
        super().__init__()
        self.ds_conv = nn.Conv2d(
            dim, dim, kernel_size=(7, 7), padding="same", groups=dim
        )
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim_out * mult, kernel_size=(3, 3), padding="same"),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, kernel_size=(3, 3), padding="same"),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.ds_conv(x)
        h = self.net(h)
        return h + self.res_conv(x)


class ConvNext(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        dim_mults: Sequence[int] = (2, 4, 8),
        depths: Sequence[int] = (3, 3, 6),
        downsampling_factors: Sequence[Sequence[int]] = ((2, 2), (2, 2), (2, 2)),
        attn: Optional[TransformerBlock] = None,
    ) -> None:
        super().__init__()
        dims = (dim, *map(lambda m: m * dim, dim_mults))
        self.attn = attn
        self.out_channels = dims[-1]
        self.stem = nn.Conv2d(1, dims[0], kernel_size=(7, 7), padding="same")
        self.layers = nn.ModuleList([])

        for i in range(len(dims) - 1):
            dim_in, dim_out = dims[i], dims[i + 1]
            self.layers.append(
                nn.ModuleList(
                    [
                        ConvNextBlock(dim_in, dim_in, 2),
                        nn.ModuleList(
                            [ConvNextBlock(dim_in, dim_in, 2) for _ in range(depths[i])]
                        ),
                        Downsample(dim_in, dim_out, downsampling_factors[i]),
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
        for init_block, blocks, down in self.layers:
            x = init_block(x)
            for fn in blocks:
                x = fn(x)
            x = down(x)
        x = self.attn(x)
        return self.norm(x)

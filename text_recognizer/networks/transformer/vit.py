"""Vision Transformer."""
from typing import Tuple, Type

from einops.layers.torch import Rearrange
import torch
from torch import nn, Tensor


class ViT(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        dim: int,
        transformer: Type[nn.Module],
        channels: int = 1,
    ) -> None:
        super().__init__()
        img_height, img_width = image_size
        patch_height, patch_width = patch_size
        assert img_height % patch_height == 0
        assert img_width % patch_width == 0

        num_patches = (img_height // patch_height) * (img_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
                c=channels,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.transformer = transformer
        self.norm = nn.LayerNorm(dim)

    def forward(self, img: Tensor) -> Tensor:
        x = self.to_patch_embedding(img)
        _, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.transformer(x)
        return x

"""Patch embedding for images and feature maps."""
from typing import Sequence, Tuple

from einops import rearrange
from loguru import logger
from torch import nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    """Patch embedding of images."""

    def __init__(
        self,
        image_shape: Sequence[int],
        patch_size: int = 16,
        in_channels: int = 1,
        embedding_dim: int = 512,
    ) -> None:
        if image_shape[0] % patch_size == 0 and image_shape[1] % patch_size == 0:
            logger.error(
                f"Image shape {image_shape} not divisable by patch size {patch_size}"
            )

        self.patch_size = patch_size
        self.embedding = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Embeds image or feature maps with patch embedding."""
        _, _, h, w = x.shape
        h_out, w_out = h // self.patch_size, w // self.patch_size
        x = self.embedding(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        return x, (h_out, w_out)

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from .embedding.sincos import sincos_2d
from .encoder import Encoder


class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]


class Vit(nn.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_height: int,
        patch_width: int,
        dim: int,
        encoder: Encoder,
        channels: int = 1,
        patch_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        patch_dim = patch_height * patch_width * channels
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h ph) (w pw) -> b (h w) (ph pw c)",
                ph=patch_height,
                pw=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.patch_embedding = sincos_2d(
            h=image_height // patch_height, w=image_width // patch_width, dim=dim
        )
        self.encoder = encoder
        self.patch_dropout = PatchDropout(patch_dropout)

    def forward(self, images: Tensor) -> Tensor:
        x = self.to_patch_embedding(images)
        x = x + self.patch_embedding.to(images.device, dtype=images.dtype)
        x = self.patch_dropout(x)
        return self.encoder(x)

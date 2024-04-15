from einops.layers.torch import Rearrange
from torch import Tensor, nn

from text_recognizer.network.convnext.convnext import ConvNext

from .transformer.embedding.sincos import sincos_2d
from .transformer.encoder import Encoder


class CVit(nn.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_height: int,
        patch_width: int,
        dim: int,
        encoder: Encoder,
        stem: ConvNext,
        channels: int = 1,
    ) -> None:
        super().__init__()
        patch_dim = patch_height * patch_width * channels
        self.stem = stem
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

    def forward(self, img: Tensor) -> Tensor:
        x = self.stem(img)
        x = self.to_patch_embedding(x)
        x += self.patch_embedding.to(img.device, dtype=img.dtype)
        return self.encoder(x)

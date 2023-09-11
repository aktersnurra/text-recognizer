from typing import Optional

from einops.layers.torch import Rearrange
from torch import Tensor, nn

from text_recognizer.network.convnext.convnext import ConvNext
from .transformer.embedding.token import TokenEmbedding
from .transformer.embedding.sincos import sincos_2d
from .transformer.decoder import Decoder
from .transformer.encoder import Encoder


class Convformer(nn.Module):
    def __init__(
        self,
        image_height: int,
        image_width: int,
        patch_height: int,
        patch_width: int,
        dim: int,
        num_classes: int,
        encoder: Encoder,
        decoder: Decoder,
        token_embedding: TokenEmbedding,
        tie_embeddings: bool,
        pad_index: int,
        stem: Optional[ConvNext] = None,
        channels: int = 1,
    ) -> None:
        super().__init__()
        patch_dim = patch_height * patch_width * channels
        self.stem = stem if stem is not None else nn.Identity()
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
        self.token_embedding = token_embedding
        self.to_logits = (
            nn.Linear(dim, num_classes)
            if not tie_embeddings
            else lambda t: t @ self.token_embedding.to_embedding.weight.t()
        )
        self.encoder = encoder
        self.decoder = decoder
        self.pad_index = pad_index

    def encode(self, img: Tensor) -> Tensor:
        x = self.stem(img)
        x = self.to_patch_embedding(x)
        x += self.patch_embedding.to(img.device, dtype=img.dtype)
        return self.encoder(x)

    def decode(self, text: Tensor, img_features: Tensor) -> Tensor:
        text = text.long()
        mask = text != self.pad_index
        tokens = self.token_embedding(text)
        output = self.decoder(tokens, context=img_features, mask=mask)
        return self.to_logits(output)

    def forward(
        self,
        img: Tensor,
        text: Tensor,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        img_features = self.encode(img)
        logits = self.decode(text, img_features)
        return logits  # [B, N, C]

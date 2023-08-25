"""Transformer module."""
from typing import Type

from einops.layers.torch import Rearrange
from torch import Tensor, nn

from text_recognizer.network.transformer.embedding.token import TokenEmbedding
from text_recognizer.network.transformer.embedding.sincos import sincos_2d
from text_recognizer.network.transformer.decoder import Decoder
from text_recognizer.network.transformer.encoder import Encoder


class VisionTransformer(nn.Module):
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
        pos_embedding: Type[nn.Module],
        tie_embeddings: bool,
        pad_index: int,
    ) -> None:
        super().__init__()
        patch_dim = patch_height * patch_width
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
        self.pos_embedding = pos_embedding
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
        x = self.to_patch_embedding(img)
        x += self.patch_embedding.to(img.device, dtype=img.dtype)
        return self.encoder(x)

    def decode(self, text: Tensor, context: Tensor) -> Tensor:
        text = text.long()
        mask = text != self.pad_index
        tokens = self.token_embedding(text)
        tokens = tokens + self.pos_embedding(tokens)
        output = self.decoder(tokens, context)
        return self.to_logits(output)

    def forward(
        self,
        img: Tensor,
        text: Tensor,
    ) -> Tensor:
        """Applies decoder block on input signals."""
        context = self.encode(img)
        logits = self.decode(text, context)
        return logits.permute(0, 2, 1)

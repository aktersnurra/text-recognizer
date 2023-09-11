from typing import Type

from einops import repeat
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from text_recognizer.network.transformer.decoder import Decoder
from text_recognizer.network.transformer.attention import Attention


class ToLatent(nn.Module):
    def __init__(self, dim: int, dim_latent: int) -> None:
        super().__init__()
        self.to_latent = nn.Linear(dim, dim_latent, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.to_latent(x), dim=-1)


class MaMMUT(nn.Module):
    def __init__(
        self,
        encoder: Type[nn.Module],
        image_attn_pool: Attention,
        decoder: Decoder,
        dim: int,
        dim_latent: int,
        num_tokens: int,
        pad_index: int,
        num_image_queries: int = 256,
    ) -> None:
        super().__init__()
        # Image encoder
        self.encoder = encoder
        self.image_queries = nn.Parameter(torch.randn(num_image_queries + 1, dim))
        self.image_attn_pool = image_attn_pool
        self.image_attn_pool_norm = nn.LayerNorm(dim)

        # Text decoder
        self.decoder = decoder
        self.pad_index = pad_index
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.text_cls_token = nn.Parameter(torch.randn(dim))
        self.text_cls_norm = nn.LayerNorm(dim)

        # To latents
        self.to_image_latent = ToLatent(dim, dim_latent)
        self.to_text_latent = ToLatent(dim, dim_latent)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_tokens, bias=False)
        )
        self.to_logits[-1].weight = self.token_embeddings.weight
        nn.init.normal_(self.token_embeddings.weight, std=0.02)

    def to_image_features(self, images: Tensor) -> Tensor:
        image_tokens = self.encoder(images)
        queries = repeat(self.image_queries, "n d -> b n d", b=image_tokens.shape[0])
        queries = self.image_attn_pool(queries, image_tokens)
        queries = self.image_attn_pool_norm(queries)
        return queries[:, 0], queries[:, 1:]

    def to_text_cls_features(self, text: Tensor) -> Tensor:
        b = text.shape[0]
        text = text.long()
        mask = text != self.pad_index
        tokens = self.token_embeddings(text)

        # Append cls token
        cls_tokens = repeat(self.text_cls_token, "d -> b 1 d", b=b)
        tokens = torch.cat([tokens, cls_tokens], dim=-2)

        # Create input mask
        mask = F.pad(mask, pad=(0, 1), value=True)

        features = self.decoder.self_attn(tokens, mask)
        cls_embedding = features[:, -1]
        return self.text_cls_norm(cls_embedding)

    def to_latents(self, image_embeddings: Tensor, text_embeddings: Tensor) -> Tensor:
        image_latent = self.to_image_latent(image_embeddings)
        text_latent = self.to_text_latent(text_embeddings)
        return image_latent, text_latent

    def decode(self, text: Tensor, image_features: Tensor) -> Tensor:
        text = text.long()
        mask = text != self.pad_index
        tokens = self.token_embeddings(text)
        text_features = self.decoder(tokens, context=image_features, mask=mask)
        return self.to_logits(text_features)

    def encode(self, images: Tensor) -> Tensor:
        _, image_features = self.to_image_features(images)
        return image_features

    def forward(self, images: Tensor, text: Tensor) -> Tensor:
        image_features = self.encode(images)
        return self.decode(text, image_features)

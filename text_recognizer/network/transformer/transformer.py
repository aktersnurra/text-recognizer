from torch import Tensor, nn

from .decoder import Decoder
from .embedding.token import TokenEmbedding
from .vit import Vit


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        encoder: Vit,
        decoder: Decoder,
        token_embedding: TokenEmbedding,
        tie_embeddings: bool,
        pad_index: int,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.to_logits = (
            nn.Linear(dim, num_classes)
            if not tie_embeddings
            else lambda t: t @ self.token_embedding.to_embedding.weight.t()
        )
        self.encoder = encoder
        self.decoder = decoder
        self.pad_index = pad_index

    def encode(self, images: Tensor) -> Tensor:
        return self.encoder(images)

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

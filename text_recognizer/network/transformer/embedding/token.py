from torch import nn, Tensor

from .l2_norm import l2_norm


class TokenEmbedding(nn.Module):
    def __init__(self, num_tokens: int, dim: int, use_l2: bool = True) -> None:
        super().__init__()
        self.use_l2 = use_l2
        self.to_embedding = nn.Embedding(num_tokens, dim)
        if self.use_l2:
            nn.init.normal_(self.to_embedding.weight, std=1e-5)
        else:
            nn.init.kaiming_normal_(self.to_embedding.weight)

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.to_embedding(x)
        return l2_norm(embedding) if self.use_l2 else embedding

"""Convolution self attention block."""

from einops import rearrange
from torch import Tensor, einsum, nn

from text_recognizer.network.convnext.norm import LayerNorm


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.fn = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, inner_dim, 1, bias=False),
            nn.GELU(),
            LayerNorm(inner_dim),
            nn.Conv2d(inner_dim, dim, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 4, dim_head: int = 64, scale: int = 8
    ) -> None:
        super().__init__()
        self.scale = scale
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) ... -> b h (...) c", h=self.heads),
            (q, k, v),
        )

        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, attn: Attention, ff: FeedForward) -> None:
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

"""Efficient self attention."""
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import einsum, nn, Tensor


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class SwiGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class Attention(nn.Module):
    def __init__(
        self, dim: int, dim_head: int = 64, heads: int = 8, mult: int = 4
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)
        attn_inner_dim = heads * dim_head
        ff_inner_dim = mult * dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (2 * ff_inner_dim))
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)
        self.ff_out = nn.Sequential(SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        h = self.heads
        x = self.norm(x)
        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        q = q * self.scale
        sim = einsum("b h i d, b j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

"""Axial self attention module."""

import torch
from torch import nn
from torch import Tensor


class SelfAttention(nn.Module):
    """Axial self attention module."""

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.dim_hidden = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, self.dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * self.dim_hidden, bias=False)
        self.to_out = nn.Linear(self.dim_hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Applies self attention."""
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        b, _, d, h, e = *q.shape, self.heads, self.dim_head

        merge_heads = (
            lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        )
        q, k, v = map(merge_heads, (q, k, v))

        energy = torch.einsum("bie,bje->bij", q, k) * (e ** -0.5)
        energy = energy.softmax(dim=-1)
        attn = torch.einsum("bij,bje->bie", energy, v)

        out = attn.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        return self.to_out(out)

"""Axial self attention module."""

import attr
import torch
from torch import nn
from torch import Tensor


@attr.s(eq=False)
class SelfAttention(nn.Module):
    """Axial self attention module."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    dim: int = attr.ib()
    dim_head: int = attr.ib()
    heads: int = attr.ib()
    dim_hidden: int = attr.ib(init=False)
    to_q: nn.Linear = attr.ib(init=False)
    to_kv: nn.Linear = attr.ib(init=False)
    to_out: nn.Linear = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self.dim_hidden = self.heads * self.dim_head
        self.to_q = nn.Linear(self.dim, self.dim_hidden, bias=False)
        self.to_kv = nn.Linear(self.dim, 2 * self.dim_hidden, bias=False)
        self.to_out = nn.Linear(self.dim_hidden, self.dim)

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

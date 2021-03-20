"""Implementes the attention module for the transformer."""
from typing import Optional, Tuple

from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Implementation of multihead attention."""

    def __init__(
        self, hidden_dim: int, num_heads: int = 8, dropout_rate: float = 0.0
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fc_q = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.fc_k = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.fc_v = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim, bias=False
        )
        self.fc_out = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        self._init_weights()

        self.dropout = nn.Dropout(p=dropout_rate)

    def _init_weights(self) -> None:
        nn.init.normal_(
            self.fc_q.weight,
            mean=0,
            std=np.sqrt(self.hidden_dim + int(self.hidden_dim / self.num_heads)),
        )
        nn.init.normal_(
            self.fc_k.weight,
            mean=0,
            std=np.sqrt(self.hidden_dim + int(self.hidden_dim / self.num_heads)),
        )
        nn.init.normal_(
            self.fc_v.weight,
            mean=0,
            std=np.sqrt(self.hidden_dim + int(self.hidden_dim / self.num_heads)),
        )
        nn.init.xavier_normal_(self.fc_out.weight)

    def scaled_dot_product_attention(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """Calculates the scaled dot product attention."""

        # Compute the energy.
        energy = torch.einsum("bhlk,bhtk->bhlt", [query, key]) / np.sqrt(
            query.shape[-1]
        )

        # If we have a mask for padding some inputs.
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -np.inf)

        # Compute the attention from the energy.
        attention = torch.softmax(energy, dim=3)

        out = torch.einsum("bhlt,bhtv->bhlv", [attention, value])
        out = rearrange(out, "b head l v -> b l (head v)")
        return out, attention

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for computing the multihead attention."""
        # Get the query, key, and value tensor.
        query = rearrange(
            self.fc_q(query), "b l (head k) -> b head l k", head=self.num_heads
        )
        key = rearrange(
            self.fc_k(key), "b t (head k) -> b head t k", head=self.num_heads
        )
        value = rearrange(
            self.fc_v(value), "b t (head v) -> b head t v", head=self.num_heads
        )

        out, attention = self.scaled_dot_product_attention(query, key, value, mask)

        out = self.fc_out(out)
        out = self.dropout(out)
        return out, attention

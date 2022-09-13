"""Fourier positional embedding."""
import numpy as np
import torch
from torch import nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Encodes a sense of distance or time for transformer networks."""

    def __init__(self, dim: int, dropout_rate: float, max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        pe = self.make_pe(dim, max_len)
        self.register_buffer("pe", pe)

    @staticmethod
    def make_pe(hidden_dim: int, max_len: int) -> Tensor:
        """Returns positional encoding."""
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-np.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Encodes the tensor with a postional embedding."""
        # [T, B, D]
        if x.shape[2] != self.pe.shape[2]:
            raise ValueError("x shape does not match pe in the 3rd dim.")
        x = x + self.pe[: x.shape[0]]
        return self.dropout(x)

"""A positional encoding for the image features, as the transformer has no notation of the order of the sequence."""
import numpy as np
import torch
from torch import nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """Encodes a sense of distance or time for transformer networks."""

    def __init__(
        self, hidden_dim: int, dropout_rate: float, max_len: int = 1000
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * -(np.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Encodes the tensor with a postional embedding."""
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)

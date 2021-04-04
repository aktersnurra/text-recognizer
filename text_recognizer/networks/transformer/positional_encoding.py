"""A positional encoding for the image features, as the transformer has no notation of the order of the sequence."""
from einops import repeat
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
        pe = self.make_pe(hidden_dim, max_len)
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
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    """Positional encodings for feature maps."""

    def __init__(self, hidden_dim: int, max_h: int = 2048, max_w: int = 2048) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"Embedding depth {hidden_dim} is not even!")
        self.hidden_dim = hidden_dim
        pe = self.make_pe(hidden_dim, max_h, max_w)
        self.register_buffer("pe", pe)

    def make_pe(hidden_dim: int, max_h: int, max_w: int) -> Tensor:
        """Returns 2d postional encoding."""
        pe_h = PositionalEncoding.make_pe(
            hidden_dim // 2, max_len=max_h
        )  # [H, 1, D // 2]
        pe_h = repeat(pe_h, "h w d -> d h (w tile)", tile=max_w)

        pe_w = PositionalEncoding.make_pe(
            hidden_dim // 2, max_len=max_h
        )  # [W, 1, D // 2]
        pe_w = repeat(pe_w, "h w d -> d (h tile) w", tile=max_h)

        pe = torch.cat([pe_h, pe_w], dim=0)  # [D, H, W]
        return pe

    def forward(self, x: Tensor) -> Tensor:
        """Adds 2D postional encoding to input tensor."""
        # Assumes x hase shape [B, D, H, W]
        if x.shape[1] != self.pe.shape[0]:
            raise ValueError("Hidden dimensions does not match.")
        x += self.pe[:, : x.shape[2], : x.shape[3]]
        return x

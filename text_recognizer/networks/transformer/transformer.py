"""Transfomer module."""
import copy
from typing import Dict, Optional, Type, Union

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from text_recognizer.networks.transformer.attention import MultiHeadAttention
from text_recognizer.networks.util import activation_function


class GEGLU(nn.Module):
    """GLU activation for improving feedforward activations."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation."""
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def _get_clones(module: Type[nn.Module], num_layers: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


class _IntraLayerConnection(nn.Module):
    """Preforms the residual connection inside the transfomer blocks and applies layernorm."""

    def __init__(self, dropout_rate: float, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, src: Tensor, residual: Tensor) -> Tensor:
        return self.norm(self.dropout(src) + residual)


class _ConvolutionalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        expansion_dim: int,
        dropout_rate: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        in_projection = (
            nn.Sequential(
                nn.Linear(hidden_dim, expansion_dim), activation_function(activation)
            )
            if activation != "glu"
            else GEGLU(hidden_dim, expansion_dim)
        )

        self.layer = nn.Sequential(
            in_projection,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=expansion_dim, out_features=hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class EncoderLayer(nn.Module):
    """Transfomer encoding layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.cnn = _ConvolutionalLayer(
            hidden_dim, expansion_dim, dropout_rate, activation
        )
        self.block1 = _IntraLayerConnection(dropout_rate, hidden_dim)
        self.block2 = _IntraLayerConnection(dropout_rate, hidden_dim)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the encoder."""
        # First block.
        # Multi head attention.
        out, _ = self.self_attention(src, src, src, mask)

        # Add & norm.
        out = self.block1(out, src)

        # Second block.
        # Apply 1D-convolution.
        cnn_out = self.cnn(out)

        # Add & norm.
        out = self.block2(cnn_out, out)

        return out


class Encoder(nn.Module):
    """Transfomer encoder module."""

    def __init__(
        self,
        num_layers: int,
        encoder_layer: Type[nn.Module],
        norm: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass through all encoder layers."""
        for layer in self.layers:
            src = layer(src, src_mask)

        if self.norm is not None:
            src = self.norm(src)

        return src


class DecoderLayer(nn.Module):
    """Transfomer decoder layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout_rate)
        self.multihead_attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout_rate
        )
        self.cnn = _ConvolutionalLayer(
            hidden_dim, expansion_dim, dropout_rate, activation
        )
        self.block1 = _IntraLayerConnection(dropout_rate, hidden_dim)
        self.block2 = _IntraLayerConnection(dropout_rate, hidden_dim)
        self.block3 = _IntraLayerConnection(dropout_rate, hidden_dim)

    def forward(
        self,
        trg: Tensor,
        memory: Tensor,
        trg_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the layer."""
        out, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.block1(out, trg)

        out, _ = self.multihead_attention(trg, memory, memory, memory_mask)
        trg = self.block2(out, trg)

        out = self.cnn(trg)
        out = self.block3(out, trg)

        return out


class Decoder(nn.Module):
    """Transfomer decoder module."""

    def __init__(
        self,
        decoder_layer: Type[nn.Module],
        num_layers: int,
        norm: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        trg: Tensor,
        memory: Tensor,
        trg_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the decoder."""
        for layer in self.layers:
            trg = layer(trg, memory, trg_mask, memory_mask)

        if self.norm is not None:
            trg = self.norm(trg)

        return trg


class Transformer(nn.Module):
    """Transformer network."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        hidden_dim: int,
        num_heads: int,
        expansion_dim: int,
        dropout_rate: float,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        # Configure encoder.
        encoder_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = EncoderLayer(
            hidden_dim, num_heads, expansion_dim, dropout_rate, activation
        )
        self.encoder = Encoder(num_encoder_layers, encoder_layer, encoder_norm)

        # Configure decoder.
        decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DecoderLayer(
            hidden_dim, num_heads, expansion_dim, dropout_rate, activation
        )
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Optional[Tensor] = None,
        trg_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the transformer."""
        if src.shape[0] != trg.shape[0]:
            print(trg.shape)
            raise RuntimeError("The batch size of the src and trg must be the same.")
        if src.shape[2] != trg.shape[2]:
            raise RuntimeError(
                "The number of features for the src and trg must be the same."
            )

        memory = self.encoder(src, src_mask)
        output = self.decoder(trg, memory, trg_mask, memory_mask)
        return output

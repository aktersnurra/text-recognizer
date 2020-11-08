"""Network with a CNN backend and a transformer encoder head."""
from typing import Dict

from einops import rearrange
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.transformer import PositionalEncoding
from text_recognizer.networks.util import configure_backbone


class CNNTransformerEncoder(nn.Module):
    """A CNN backbone with Transformer Encoder frontend for sequence prediction."""

    def __init__(
        self,
        backbone: str,
        backbone_args: Dict,
        mlp_dim: int,
        d_model: int,
        nhead: int = 8,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        num_layers: int = 6,
        num_classes: int = 80,
        num_channels: int = 256,
        max_len: int = 97,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.num_layers = num_layers

        self.backbone = configure_backbone(backbone, backbone_args)
        self.position_encoding = PositionalEncoding(d_model, dropout_rate)
        self.encoder = self._configure_encoder()

        self.conv = nn.Conv2d(num_channels, max_len, kernel_size=1)

        self.mlp = nn.Linear(mlp_dim, d_model)

        self.head = nn.Linear(d_model, num_classes)

    def _configure_encoder(self) -> nn.TransformerEncoder:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout_rate,
            activation=self.activation,
        )
        norm = nn.LayerNorm(self.d_model)
        return nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.num_layers, norm=norm
        )

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        """Forward pass through the network."""
        if len(x.shape) < 4:
            x = x[(None,) * (4 - len(x.shape))]

        x = self.conv(self.backbone(x))
        x = rearrange(x, "b c h w -> b c (h w)")
        x = self.mlp(x)
        x = self.position_encoding(x)
        x = rearrange(x, "b c h-> c b h")
        x = self.encoder(x)
        x = rearrange(x, "c b h-> b c h")
        logits = self.head(x)

        return logits

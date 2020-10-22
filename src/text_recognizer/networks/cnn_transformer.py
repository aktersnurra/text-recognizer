"""A DETR style transfomers but for text recognition."""
from typing import Dict, Optional, Tuple, Type

from einops.layers.torch import Rearrange
from loguru import logger
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.transformer import PositionalEncoding, Transformer
from text_recognizer.networks.util import configure_backbone


class CNNTransformer(nn.Module):
    """CNN+Transfomer for image to sequence prediction, sort of based on the ideas from DETR."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int,
        max_len: int,
        expansion_dim: int,
        dropout_rate: float,
        trg_pad_index: int,
        backbone: str,
        backbone_args: Optional[Dict] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.trg_pad_index = trg_pad_index
        self.backbone_args = backbone_args
        self.backbone = configure_backbone(backbone, backbone_args)
        self.character_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_encoding = PositionalEncoding(hidden_dim, dropout_rate, max_len)
        self.collapse_spatial_dim = nn.Sequential(
            Rearrange("b t h w -> b t (h w)"), nn.AdaptiveAvgPool2d((None, hidden_dim))
        )
        self.transformer = Transformer(
            num_encoder_layers,
            num_decoder_layers,
            hidden_dim,
            num_heads,
            expansion_dim,
            dropout_rate,
            activation,
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def _create_trg_mask(self, trg: Tensor) -> Tensor:
        # Move this outside the transformer.
        trg_pad_mask = (trg != self.trg_pad_index)[:, None, None]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=trg.device)
        ).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def encoder(self, src: Tensor) -> Tensor:
        """Forward pass with the encoder of the transformer."""
        return self.transformer.encoder(src)

    def decoder(self, trg: Tensor, memory: Tensor, trg_mask: Tensor) -> Tensor:
        """Forward pass with the decoder of the transformer + classification head."""
        return self.head(
            self.transformer.decoder(trg=trg, memory=memory, trg_mask=trg_mask)
        )

    def preprocess_input(self, src: Tensor) -> Tensor:
        """Encodes src with a backbone network and a positional encoding.

        Args:
            src (Tensor): Input tensor.

        Returns:
            Tensor: A input src to the transformer.

        """
        # If batch dimenstion is missing, it needs to be added.
        if len(src.shape) < 4:
            src = src[(None,) * (4 - len(src.shape))]
        src = self.backbone(src)
        src = self.collapse_spatial_dim(src)
        src = self.position_encoding(src)
        return src

    def preprocess_target(self, trg: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes target tensor with embedding and postion.

        Args:
            trg (Tensor): Target tensor.

        Returns:
            Tuple[Tensor, Tensor]: Encoded target tensor and target mask.

        """
        trg_mask = self._create_trg_mask(trg)
        trg = self.character_embedding(trg.long())
        trg = self.position_encoding(trg)
        return trg, trg_mask

    def forward(self, x: Tensor, trg: Tensor) -> Tensor:
        """Forward pass with CNN transfomer."""
        src = self.preprocess_input(x)
        trg, trg_mask = self.preprocess_target(trg)
        out = self.transformer(src, trg, trg_mask=trg_mask)
        logits = self.head(out)
        return logits

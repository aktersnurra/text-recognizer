"""A CNN-Transformer for image to text recognition."""
from typing import Dict, Optional, Tuple

from einops import rearrange
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.transformer import PositionalEncoding, Transformer
from text_recognizer.networks.util import configure_backbone


class CNNTransformer(nn.Module):
    """CNN+Transfomer for image to sequence prediction."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int,
        adaptive_pool_dim: Tuple,
        expansion_dim: int,
        dropout_rate: float,
        trg_pad_index: int,
        backbone: str,
        backbone_args: Optional[Dict] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.trg_pad_index = trg_pad_index
        self.backbone = configure_backbone(backbone, backbone_args)
        self.character_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_encoding = PositionalEncoding(hidden_dim, dropout_rate)

        self.adaptive_pool = (
            nn.AdaptiveAvgPool2d((adaptive_pool_dim)) if adaptive_pool_dim else None
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

        self.head = nn.Sequential(nn.Linear(hidden_dim, vocab_size),)

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

    def extract_image_features(self, src: Tensor) -> Tensor:
        """Extracts image features with a backbone neural network.

        It seem like the winning idea was to swap channels and width dimension and collapse
        the height dimension. The transformer is learning like a baby with this implementation!!! :D
        Ohhhh, the joy I am experiencing right now!! Bring in the beers! :D :D :D

        Args:
            src (Tensor): Input tensor.

        Returns:
            Tensor: A input src to the transformer.

        """
        # If batch dimension is missing, it needs to be added.
        if len(src.shape) < 4:
            src = src[(None,) * (4 - len(src.shape))]
        src = self.backbone(src)

        if self.adaptive_pool is not None:
            src = rearrange(src, "b c h w -> b w c h")
            src = self.adaptive_pool(src)
            src = src.squeeze(3)
        else:
            src = rearrange(src, "b c h w -> b (w h) c")

        src = self.position_encoding(src)

        return src

    def target_embedding(self, trg: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes target tensor with embedding and postion.

        Args:
            trg (Tensor): Target tensor.

        Returns:
            Tuple[Tensor, Tensor]: Encoded target tensor and target mask.

        """
        trg = self.character_embedding(trg.long())
        trg = self.position_encoding(trg)
        return trg

    def decode_image_features(self, h: Tensor, trg: Optional[Tensor] = None) -> Tensor:
        """Takes images features from the backbone and decodes them with the transformer."""
        trg_mask = self._create_trg_mask(trg)
        trg = self.target_embedding(trg)
        out = self.transformer(h, trg, trg_mask=trg_mask)

        logits = self.head(out)
        return logits

    def forward(self, x: Tensor, trg: Optional[Tensor] = None) -> Tensor:
        """Forward pass with CNN transfomer."""
        h = self.extract_image_features(x)
        logits = self.decode_image_features(h, trg)
        return logits

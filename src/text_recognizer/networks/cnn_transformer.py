"""A DETR style transfomers but for text recognition."""
from typing import Dict, Optional, Tuple

from einops import rearrange
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
        adaptive_pool_dim: Tuple,
        expansion_dim: int,
        dropout_rate: float,
        trg_pad_index: int,
        backbone: str,
        out_channels: int,
        max_len: int,
        backbone_args: Optional[Dict] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.trg_pad_index = trg_pad_index

        self.backbone = configure_backbone(backbone, backbone_args)
        self.character_embedding = nn.Embedding(vocab_size, hidden_dim)

        # self.conv = nn.Conv2d(out_channels, max_len, kernel_size=1)

        self.position_encoding = PositionalEncoding(hidden_dim, dropout_rate)
        self.row_embed = nn.Parameter(torch.rand(max_len, max_len // 2))
        self.col_embed = nn.Parameter(torch.rand(max_len, max_len // 2))

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
        # src = self.conv(src)
        if self.adaptive_pool is not None:
            src = self.adaptive_pool(src)
        H, W = src.shape[-2:]
        src = rearrange(src, "b t h w -> b t (h w)")

        # construct positional encodings
        pos = torch.cat(
            [
                self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
            ],
            dim=-1,
        ).unsqueeze(0)
        pos = rearrange(pos, "b h w l -> b l (h w)")
        src = pos + 0.1 * src
        return src

    def preprocess_target(self, trg: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes target tensor with embedding and postion.

        Args:
            trg (Tensor): Target tensor.

        Returns:
            Tuple[Tensor, Tensor]: Encoded target tensor and target mask.

        """
        trg = self.character_embedding(trg.long())
        trg = self.position_encoding(trg)
        return trg

    def forward(self, x: Tensor, trg: Optional[Tensor] = None) -> Tensor:
        """Forward pass with CNN transfomer."""
        h = self.preprocess_input(x)
        trg_mask = self._create_trg_mask(trg)
        trg = self.preprocess_target(trg)
        out = self.transformer(h, trg, trg_mask=trg_mask)

        logits = self.head(out)
        return logits

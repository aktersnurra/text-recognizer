"""A VQ-Transformer for image to text recognition."""
from typing import Dict, Optional, Tuple

from einops import rearrange, repeat
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.transformer import PositionalEncoding, Transformer
from text_recognizer.networks.util import activation_function
from text_recognizer.networks.util import configure_backbone
from text_recognizer.networks.vqvae.encoder import _ResidualBlock


class VQTransformer(nn.Module):
    """VQ+Transfomer for image to character sequence prediction."""

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
        max_len: int,
        backbone: str,
        backbone_args: Optional[Dict] = None,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        # Configure vector quantized backbone.
        self.backbone = configure_backbone(backbone, backbone_args)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        # Configure embeddings for Transformer network.
        self.trg_pad_index = trg_pad_index
        self.vocab_size = vocab_size
        self.character_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.src_position_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        self.trg_position_encoding = PositionalEncoding(hidden_dim, dropout_rate)
        nn.init.normal_(self.character_embedding.weight, std=0.02)

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

    def extract_image_features(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """Extracts image features with a backbone neural network.

        It seem like the winning idea was to swap channels and width dimension and collapse
        the height dimension. The transformer is learning like a baby with this implementation!!! :D
        Ohhhh, the joy I am experiencing right now!! Bring in the beers! :D :D :D

        Args:
            src (Tensor): Input tensor.

        Returns:
            Tensor: The input src to the transformer and the vq loss.

        """
        # If batch dimension is missing, it needs to be added.
        if len(src.shape) < 4:
            src = src[(None,) * (4 - len(src.shape))]
        src, vq_loss = self.backbone.encode(src)
        # src = self.backbone.decoder.res_block(src)
        src = self.conv(src)

        if self.adaptive_pool is not None:
            src = rearrange(src, "b c h w -> b w c h")
            src = self.adaptive_pool(src)
            src = src.squeeze(3)
        else:
            src = rearrange(src, "b c h w -> b (w h) c")

        b, t, _ = src.shape

        src += self.src_position_embedding[:, :t]

        return src, vq_loss

    def target_embedding(self, trg: Tensor) -> Tensor:
        """Encodes target tensor with embedding and postion.

        Args:
            trg (Tensor): Target tensor.

        Returns:
            Tensor: Encoded target tensor.

        """
        trg = self.character_embedding(trg.long())
        trg = self.trg_position_encoding(trg)
        return trg

    def decode_image_features(
        self, image_features: Tensor, trg: Optional[Tensor] = None
    ) -> Tensor:
        """Takes images features from the backbone and decodes them with the transformer."""
        trg_mask = self._create_trg_mask(trg)
        trg = self.target_embedding(trg)
        out = self.transformer(image_features, trg, trg_mask=trg_mask)

        logits = self.head(out)
        return logits

    def forward(self, x: Tensor, trg: Optional[Tensor] = None) -> Tensor:
        """Forward pass with CNN transfomer."""
        image_features, vq_loss = self.extract_image_features(x)
        logits = self.decode_image_features(image_features, trg)
        return logits, vq_loss

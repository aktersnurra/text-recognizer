"""A Vision Transformer.

Inspired by:
https://openreview.net/pdf?id=YicbFdNTTy

"""
from typing import Optional, Tuple

from einops import rearrange, repeat
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.transformer import Transformer


class ViT(nn.Module):
    """Transfomer for image to sequence prediction."""

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int,
        expansion_dim: int,
        patch_dim: Tuple[int, int],
        image_size: Tuple[int, int],
        dropout_rate: float,
        trg_pad_index: int,
        max_len: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.trg_pad_index = trg_pad_index
        self.patch_dim = patch_dim
        self.num_patches = image_size[-1] // self.patch_dim[1]

        # Encoder
        self.patch_to_embedding = nn.Linear(
            self.patch_dim[0] * self.patch_dim[1], hidden_dim
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.character_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        self.dropout = nn.Dropout(dropout_rate)
        self._init()

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

    def _init(self) -> None:
        nn.init.normal_(self.character_embedding.weight, std=0.02)
        # nn.init.normal_(self.pos_embedding.weight, std=0.02)

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

        patches = rearrange(
            src,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_dim[0],
            p2=self.patch_dim[1],
        )

        # From patches to encoded sequence.
        x = self.patch_to_embedding(patches)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        return x

    def target_embedding(self, trg: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes target tensor with embedding and postion.

        Args:
            trg (Tensor): Target tensor.

        Returns:
            Tuple[Tensor, Tensor]: Encoded target tensor and target mask.

        """
        _, n = trg.shape
        trg = self.character_embedding(trg.long())
        trg += self.pos_embedding[:, :n]
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

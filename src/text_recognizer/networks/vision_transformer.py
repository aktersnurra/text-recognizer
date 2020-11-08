"""VisionTransformer module.

Splits each image into patches and feeds them to a transformer.

"""

from typing import Dict, Optional, Tuple, Type

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from loguru import logger
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.transformer import PositionalEncoding, Transformer
from text_recognizer.networks.util import configure_backbone


class VisionTransformer(nn.Module):
    """Linear projection+Transfomer for image to sequence prediction, sort of based on the ideas from ViT."""

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
        mlp_dim: Optional[int] = None,
        patch_size: Tuple[int, int] = (28, 28),
        stride: Tuple[int, int] = (1, 14),
        activation: str = "gelu",
        backbone: Optional[str] = None,
        backbone_args: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride
        self.trg_pad_index = trg_pad_index
        self.slidning_window = self._configure_sliding_window()
        self.character_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_encoding = PositionalEncoding(hidden_dim, dropout_rate, max_len)
        self.mlp_dim = mlp_dim

        self.use_backbone = False
        if backbone is None:
            self.linear_projection = nn.Linear(
                self.patch_size[0] * self.patch_size[1], hidden_dim
            )
        else:
            self.backbone = configure_backbone(backbone, backbone_args)
            if mlp_dim:
                self.mlp = nn.Linear(mlp_dim, hidden_dim)
            self.use_backbone = True

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

    def _configure_sliding_window(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Unfold(kernel_size=self.patch_size, stride=self.stride),
            Rearrange(
                "b (c h w) t -> b t c h w",
                h=self.patch_size[0],
                w=self.patch_size[1],
                c=1,
            ),
        )

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

    def _backbone(self, x: Tensor) -> Tensor:
        b, t = x.shape[:2]
        if self.use_backbone:
            x = rearrange(x, "b t c h w -> (b t) c h w", b=b, t=t)
            x = self.backbone(x)
            if self.mlp_dim:
                x = rearrange(x, "(b t) c h w -> b t (c h w)", b=b, t=t)
                x = self.mlp(x)
            else:
                x = rearrange(x, "(b t) h -> b t h", b=b, t=t)
        else:
            x = rearrange(x, "b t c h w -> b t (c h w)", b=b, t=t)
            x = self.linear_projection(x)
        return x

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
        src = self.slidning_window(src)  # .squeeze(-2)
        src = self._backbone(src)
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
        """Forward pass with vision transfomer."""
        src = self.preprocess_input(x)
        trg, trg_mask = self.preprocess_target(trg)
        out = self.transformer(src, trg, trg_mask=trg_mask)
        logits = self.head(out)
        return logits

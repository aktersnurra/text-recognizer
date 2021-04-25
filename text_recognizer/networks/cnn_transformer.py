"""A Transformer with a cnn backbone.

The network encodes a image with a convolutional backbone to a latent representation,
i.e. feature maps. A 2d positional encoding is applied to the feature maps for
spatial information. The resulting feature are then set to a transformer decoder
together with the target tokens.

TODO: Local attention for lower layer in attention.

"""
import importlib
import math
from typing import Dict, Optional, Union, Sequence, Type

from einops import rearrange
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch import Tensor

from text_recognizer.data.emnist import NUM_SPECIAL_TOKENS
from text_recognizer.networks.transformer import (
    Decoder,
    DecoderLayer,
    PositionalEncoding,
    PositionalEncoding2D,
    target_padding_mask,
)

NUM_WORD_PIECES = 1000


class CNNTransformer(nn.Module):
    def __init__(
        self,
        input_dim: Sequence[int],
        output_dims: Sequence[int],
        encoder: Union[DictConfig, Dict],
        vocab_size: Optional[int] = None,
        num_decoder_layers: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 4,
        expansion_dim: int = 1024,
        dropout_rate: float = 0.1,
        transformer_activation: str = "glu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.vocab_size = (
            NUM_WORD_PIECES + NUM_SPECIAL_TOKENS if vocab_size is None else vocab_size
        )
        self.pad_index = 3  # TODO: fix me
        self.hidden_dim = hidden_dim
        self.max_output_length = output_dims[0]

        # Image backbone
        self.encoder = self._configure_encoder(encoder)
        self.encoder_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.feature_map_encoding = PositionalEncoding2D(
            hidden_dim=hidden_dim, max_h=input_dim[1], max_w=input_dim[2]
        )

        # Target token embedding
        self.trg_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.trg_position_encoding = PositionalEncoding(
            hidden_dim, dropout_rate, max_len=output_dims[0]
        )

        # Transformer decoder
        self.decoder = Decoder(
            decoder_layer=DecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                expansion_dim=expansion_dim,
                dropout_rate=dropout_rate,
                activation=transformer_activation,
            ),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # Classification head
        self.head = nn.Linear(hidden_dim, self.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        self.trg_embedding.weight.data.uniform_(-0.1, 0.1)
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-0.1, 0.1)

        nn.init.kaiming_normal_(
            self.encoder_proj.weight.data,
            a=0,
            mode="fan_out",
            nonlinearity="relu",
        )
        if self.encoder_proj.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(
                self.encoder_proj.weight.data
            )
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.encoder_proj.bias, -bound, bound)

    @staticmethod
    def _configure_encoder(encoder: Union[DictConfig, Dict]) -> Type[nn.Module]:
        encoder = OmegaConf.create(encoder)
        args = encoder.args or {}
        network_module = importlib.import_module("text_recognizer.networks")
        encoder_class = getattr(network_module, encoder.type)
        return encoder_class(**args)

    def encode(self, image: Tensor) -> Tensor:
        """Extracts image features with backbone.

        Args:
            image (Tensor): Image(s) of handwritten text.

        Retuns:
            Tensor: Image features.

        Shapes:
            - image: :math: `(B, C, H, W)`
            - latent: :math: `(B, T, C)`

        """
        # Extract image features.
        image_features = self.encoder(image)
        image_features = self.encoder_proj(image_features)

        # Add 2d encoding to the feature maps.
        image_features = self.feature_map_encoding(image_features)

        # Collapse features maps height and width.
        image_features = rearrange(image_features, "b c h w -> b (h w) c")
        return image_features

    def decode(self, memory: Tensor, trg: Tensor) -> Tensor:
        """Decodes image features with transformer decoder."""
        trg_mask = target_padding_mask(trg=trg, pad_index=self.pad_index)
        trg = self.trg_embedding(trg) * math.sqrt(self.hidden_dim)
        trg = rearrange(trg, "b t d -> t b d")
        trg = self.trg_position_encoding(trg)
        trg = rearrange(trg, "t b d -> b t d")
        out = self.decoder(trg=trg, memory=memory, trg_mask=trg_mask, memory_mask=None)
        logits = self.head(out)
        return logits

    def forward(self, image: Tensor, trg: Tensor) -> Tensor:
        image_features = self.encode(image)
        output = self.decode(image_features, trg)
        output = rearrange(output, "b t c -> b c t")
        return output

    def predict(self, image: Tensor) -> Tensor:
        """Transcribes text in image(s)."""
        bsz = image.shape[0]
        image_features = self.encode(image)

        output_tokens = (
            (torch.ones((bsz, self.max_output_length)) * self.pad_index)
            .type_as(image)
            .long()
        )
        output_tokens[:, 0] = self.start_index
        for i in range(1, self.max_output_length):
            trg = output_tokens[:, :i]
            output = self.decode(image_features, trg)
            output = torch.argmax(output, dim=-1)
            output_tokens[:, i] = output[-1:]

        # Set all tokens after end token to be padding.
        for i in range(1, self.max_output_length):
            indices = output_tokens[:, i - 1] == self.end_index | (
                output_tokens[:, i - 1] == self.pad_index
            )
            output_tokens[indices, i] = self.pad_index

        return output_tokens

from typing import Optional, Tuple, Type

from torch import nn, Tensor

from text_recognizer.networks.transformer.decoder import Decoder
from text_recognizer.networks.transformer.embeddings.axial import (
    AxialPositionalEmbedding,
)

from text_recognizer.networks.conv_transformer import ConvTransformer
from text_recognizer.networks.quantizer.quantizer import VectorQuantizer


class VqTransformer(ConvTransformer):
    def __init__(
        self,
        input_dims: Tuple[int, int, int],
        hidden_dim: int,
        num_classes: int,
        pad_index: Tensor,
        encoder: Type[nn.Module],
        decoder: Decoder,
        pixel_embedding: AxialPositionalEmbedding,
        token_pos_embedding: Optional[Type[nn.Module]] = None,
        quantizer: Optional[VectorQuantizer] = None,
    ) -> None:
        super().__init__(
            input_dims,
            hidden_dim,
            num_classes,
            pad_index,
            encoder,
            decoder,
            pixel_embedding,
            token_pos_embedding,
        )
        self.quantizer = quantizer

    def quantize(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        q, _, loss = self.quantizer(z)
        return q, loss

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        z = self.conv(z)
        q, loss = self.quantize(z)
        z = self.pixel_embedding(q)
        z = z.flatten(start_dim=2)

        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z, loss

    def forward(self, x: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        z, loss = self.encode(x)
        logits = self.decode(z, context)
        return logits, loss

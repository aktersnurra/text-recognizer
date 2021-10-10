"""Vector quantized encoder, transformer decoder."""
from pathlib import Path
from typing import OrderedDict, Tuple

from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
from torch import Tensor

from text_recognizer.networks.vqvae.vqvae import VQVAE
from text_recognizer.networks.conv_transformer import ConvTransformer
from text_recognizer.networks.transformer.layers import Decoder


class VqTransformer(ConvTransformer):
    """Convolutional encoder and transformer decoder network."""

    def __init__(
        self,
        input_dims: Tuple[int, int, int],
        encoder_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        num_classes: int,
        pad_index: Tensor,
        decoder: Decoder,
        no_grad: bool,
        pretrained_encoder_path: str,
    ) -> None:
        # For typing
        self.encoder: VQVAE = None
        self.no_grad = no_grad

        super().__init__(
            input_dims=input_dims,
            encoder_dim=encoder_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
            pad_index=pad_index,
            encoder=self.encoder,
            decoder=decoder,
        )
        self._setup_encoder(pretrained_encoder_path)

    def _load_state_dict(self, path: Path) -> OrderedDict:
        weights_path = list((path / "checkpoints").glob("epoch=*.ckpt"))[0]
        renamed_state_dict = OrderedDict()
        state_dict = torch.load(weights_path)["state_dict"]
        for key in state_dict.keys():
            if "network" in key:
                new_key = key.removeprefix("network.")
                renamed_state_dict[new_key] = state_dict[key]
        del state_dict
        return renamed_state_dict

    def _setup_encoder(self, pretrained_encoder_path: str,) -> None:
        """Load encoder module."""
        path = Path(__file__).resolve().parents[2] / pretrained_encoder_path
        with open(path / "config.yaml") as f:
            cfg = OmegaConf.load(f)
        state_dict = self._load_state_dict(path)
        self.encoder = instantiate(cfg.network)
        self.encoder.load_state_dict(state_dict)
        del self.encoder.decoder

    def _encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z_e = self.encoder.encode(x)
        z_q, commitment_loss = self.encoder.quantize(z_e)
        z = self.encoder.post_codebook_conv(z_q)
        return z, commitment_loss

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes an image into a discrete (VQ) latent representation.

        Args:
            x (Tensor): Image tensor.

        Shape:
            - x: :math: `(B, C, H, W)`
            - z: :math: `(B, Sx, E)`

            where Sx is the length of the flattened feature maps projected from
            the encoder. E latent dimension for each pixel in the projected
            feature maps.

        Returns:
            Tensor: A Latent embedding of the image.
        """
        if self.no_grad:
            with torch.no_grad():
                z_q, commitment_loss = self._encode(x)
        else:
            z_q, commitment_loss = self._encode(x)

        z = self.latent_encoder(z_q)

        # Permute tensor from [B, E, Ho * Wo] to [B, Sx, E]
        z = z.permute(0, 2, 1)
        return z, commitment_loss

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """Encodes images into word piece logtis.

        Args:
            x (Tensor): Input image(s).
            context (Tensor): Target word embeddings.

        Shapes:
            - x: :math: `(B, C, H, W)`
            - context: :math: `(B, Sy, T)`

            where B is the batch size, C is the number of input channels, H is
            the image height and W is the image width.

        Returns:
            Tensor: Sequence of logits.
        """
        z, commitment_loss = self.encode(x)
        logits = self.decode(z, context)
        return logits, commitment_loss

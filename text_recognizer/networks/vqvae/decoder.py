"""CNN decoder for the VQ-VAE."""
import attr
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function
from text_recognizer.networks.vqvae.residual import Residual


@attr.s(eq=False)
class Decoder(nn.Module):
    """A CNN encoder network."""

    in_channels: int = attr.ib()
    embedding_dim: int = attr.ib()
    out_channels: int = attr.ib()
    res_channels: int = attr.ib()
    num_residual_layers: int = attr.ib()
    activation: str = attr.ib()
    decoder: nn.Sequential = attr.ib(init=False)

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        self.decoder = self._build_decompression_block()

    def _build_decompression_block(self,) -> nn.Sequential:
        activation_fn = activation_function(self.activation)
        blocks = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embedding_dim,
                kernel_size=3,
                padding=1,
            )
        ]
        for _ in range(self.num_residual_layers):
            blocks.append(
                Residual(in_channels=self.embedding_dim, out_channels=self.res_channels)
            )
        blocks.append(activation_fn)
        blocks += [
            nn.ConvTranspose2d(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation_fn,
            nn.ConvTranspose2d(
                in_channels=self.embedding_dim // 2,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        ]
        return nn.Sequential(*blocks)

    def forward(self, z_q: Tensor) -> Tensor:
        """Reconstruct input from given codes."""
        return self.decoder(z_q)

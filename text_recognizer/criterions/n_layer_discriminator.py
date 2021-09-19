"""Pix2pix discriminator loss."""
from torch import nn, Tensor

from text_recognizer.networks.vqvae.norm import Normalize


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator loss in Pix2Pix."""

    def __init__(
        self, in_channels: int = 1, num_channels: int = 32, num_layers: int = 3
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.discriminator = self._build_discriminator()

    def _build_discriminator(self) -> nn.Sequential:
        """Builds discriminator."""
        discriminator = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.num_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Mish(inplace=True),
        ]
        in_channels = self.num_channels
        for n in range(1, self.num_layers):
            discriminator += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels * n,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                # Normalize(num_channels=in_channels * n),
                nn.Mish(inplace=True),
            ]
            in_channels *= n

        discriminator += [
            nn.Conv2d(
                in_channels=self.num_channels * (self.num_layers - 1),
                out_channels=1,
                kernel_size=4,
                padding=1,
            )
        ]
        return nn.Sequential(*discriminator)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through discriminator."""
        return self.discriminator(x)

"""Residual block."""
import attr
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import activation_function
from text_recognizer.networks.vqvae.norm import Normalize


@attr.s(eq=False)
class Residual(nn.Module):
    in_channels: int = attr.ib()
    residual_channels: int = attr.ib()
    use_norm: bool = attr.ib(default=False)
    activation: str = attr.ib(default="relu")

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        super().__init__()
        self.block = self._build_res_block()

    def _build_res_block(self) -> nn.Sequential:
        """Build residual block."""
        block = []
        activation_fn = activation_function(activation=self.activation)

        if self.use_norm:
            block.append(Normalize(num_channels=self.in_channels))

        block += [
            activation_fn,
            nn.Conv2d(
                self.in_channels,
                self.residual_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        ]

        if self.use_norm:
            block.append(Normalize(num_channels=self.residual_channels))

        block += [
            activation_fn,
            nn.Conv2d(
                self.residual_channels, self.in_channels, kernel_size=1, bias=False
            ),
        ]
        return nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual forward pass."""
        return x + self.block(x)

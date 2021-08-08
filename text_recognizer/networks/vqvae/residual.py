"""Residual block."""
import attr
from torch import nn
from torch import Tensor

from text_recognizer.networks.vqvae.norm import Normalize


@attr.s(eq=False)
class Residual(nn.Module):
    in_channels: int = attr.ib()
    out_channels: int = attr.ib()
    dropout_rate: float = attr.ib(default=0.0)
    use_norm: bool = attr.ib(default=False)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        super().__init__()
        self.block = self._build_res_block()
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_shortcut = None

    def _build_res_block(self) -> nn.Sequential:
        """Build residual block."""
        block = []
        if self.use_norm:
            block.append(Normalize(num_channels=self.in_channels))
        block += [
            nn.Mish(),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        ]
        if self.dropout_rate:
            block += [nn.Dropout(p=self.dropout_rate)]

        if self.use_norm:
            block.append(Normalize(num_channels=self.out_channels))

        block += [
            nn.Mish(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
        ]
        return nn.Sequential(*block)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual forward pass."""
        residual = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return residual + self.block(x)

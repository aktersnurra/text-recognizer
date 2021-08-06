"""Normalizer block."""
import attr
from torch import nn, Tensor


@attr.s(eq=False)
class Normalize(nn.Module):
    num_channels: int = attr.ib()
    norm: nn.GroupNorm = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=self.num_channels, num_channels=self.num_channels, eps=1.0e-6, affine=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies group normalization."""
        return self.norm(x)

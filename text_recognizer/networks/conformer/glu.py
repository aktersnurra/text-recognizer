"""GLU layer."""
from torch import nn, Tensor


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

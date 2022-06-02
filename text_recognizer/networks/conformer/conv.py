"""Conformer convolutional block."""
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, Tensor


from text_recognizer.networks.conformer.depth_wise_conv import DepthwiseConv1D
from text_recognizer.networks.conformer.glu import GLU


class ConformerConv(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 2,
        kernel_size: int = 31,
        dropout: int = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = expansion_factor * dim
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1D(dim, 2 * inner_dim, 1),
            GLU(dim=1),
            DepthwiseConv1D(inner_dim, inner_dim, kernel_size),
            nn.BatchNorm1d(inner_dim),
            nn.Mish(inplace=True),
            nn.Conv1D(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

"""Axial transformer encoder."""

from typing import List, Optional, Type
from text_recognizer.networks.transformer.embeddings.axial import (
    AxialPositionalEmbeddingImage,
)

from torch import nn, Tensor

from text_recognizer.networks.transformer.axial_attention.self_attention import (
    SelfAttention,
)
from text_recognizer.networks.transformer.axial_attention.utils import (
    calculate_permutations,
    PermuteToForm,
    Sequential,
)
from text_recognizer.networks.transformer.norm import PreNorm


class AxialEncoder(nn.Module):
    """Axial transfomer encoder."""

    def __init__(
        self,
        shape: List[int],
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        dim_index: int,
        axial_embedding: AxialPositionalEmbeddingImage,
    ) -> None:
        super().__init__()

        self.shape = shape
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.dim_index = dim_index
        self.axial_embedding = axial_embedding

        self.fn = self._build()

    def _build(self) -> Sequential:
        permutations = calculate_permutations(2, self.dim_index)
        get_ff = lambda: nn.Sequential(
            nn.LayerNorm([self.dim, *self.shape]),
            nn.Conv2d(
                in_channels=self.dim,
                out_channels=4 * self.dim,
                kernel_size=3,
                padding=1,
            ),
            nn.Mish(inplace=True),
            nn.Conv2d(
                in_channels=4 * self.dim,
                out_channels=self.dim,
                kernel_size=3,
                padding=1,
            ),
        )

        layers = nn.ModuleList([])
        for _ in range(self.depth):
            attns = nn.ModuleList(
                [
                    PermuteToForm(
                        permutation=permutation,
                        fn=PreNorm(
                            self.dim,
                            SelfAttention(
                                dim=self.dim, heads=self.heads, dim_head=self.dim_head
                            ),
                        ),
                    )
                    for permutation in permutations
                ]
            )
            convs = nn.ModuleList([get_ff(), get_ff()])
            layers.append(attns)
            layers.append(convs)

        return Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Applies fn to input."""
        x += self.axial_embedding(x)
        return self.fn(x)

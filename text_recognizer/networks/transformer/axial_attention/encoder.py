"""Axial transformer encoder."""

from typing import List

import attr
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


@attr.s(eq=False)
class AxialEncoder(nn.Module):
    """Axial transfomer encoder."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    shape: List[int] = attr.ib()
    dim: int = attr.ib()
    depth: int = attr.ib()
    heads: int = attr.ib()
    dim_head: int = attr.ib()
    dim_index: int = attr.ib()
    fn: nn.Sequential = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        self._build()

    def _build(self) -> None:
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

        self.fn = Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        """Applies fn to input."""
        return self.fn(x)

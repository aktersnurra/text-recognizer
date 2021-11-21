"""Helper functions for axial attention."""
from operator import itemgetter
from typing import Callable, List, Tuple

import attr
from torch import nn, Tensor


def _map_el_ind(arr: Tensor, ind: int) -> List:
    return list(map(itemgetter(ind), arr))


def _sort_indices(arr: Tensor) -> Tuple[List[int], List[int]]:
    indices = [i for i in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return _map_el_ind(arr, 0), _map_el_ind(arr, 1)


def calculate_permutations(num_dims: int, emb_dim: int) -> List[List[int]]:
    """Returns permutations of tensor."""
    total_dims = num_dims + 2
    axial_dims = [i for i in range(1, total_dims) if i != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dims)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


@attr.s(eq=False)
class PermuteToForm(nn.Module):
    """Helper class for applying axial attention."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    fn: Callable = attr.ib()
    permutation: List[List[int]] = attr.ib()
    inv_permutation: List[List[int]] = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        _, self.inv_permutation = _sort_indices(self.permutation)

    def forward(self, x: Tensor) -> Tensor:
        """Permutes tensor, applies axial attention, permutes tensor back."""
        x = x.permute(*self.permutation).contiguous()
        shape = x.shape
        *_, t, d = shape

        # Merge all but axial dimension
        x = x.reshape(-1, t, d)

        # Apply attention
        x = self.fn(x)

        # Restore original shape and permutation
        x = x.reshape(*shape)
        x = x.permute(*self.inv_permutation).contiguous()
        return x


class Sequential(nn.Module):
    """Applies a list of paired functions to input."""

    def __init__(self, fns: nn.ModuleList) -> None:
        super().__init__()
        self.fns = fns

    def forward(self, x: Tensor) -> Tensor:
        """Applies blocks to input."""
        for f, g in self.fns:
            x = x + f(x)
            x = x + g(x)
        return x

from typing import Union

from torch import Tensor


def first_element(x: Tensor, element: Union[int, float], dim: int = 1) -> Tensor:
    nonz = x == element
    ind = ((nonz.cumsum(dim) == 1) & nonz).max(dim).indices
    ind[ind == 0] = x.shape[dim]
    return ind

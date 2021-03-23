"""Base PyTorch Dataset class."""
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base Dataset class that processes data and targets through optional transfroms.

    Args:
        data (Union[Sequence, Tensor]): Torch tensors, numpy arrays, or PIL images.
        targets (Union[Sequence, Tensor]): Torch tensors or numpy arrays.
        tranform (Callable): Function that takes a datum and applies transforms.
        target_transform (Callable): Fucntion that takes a target and applies
            target transforms.
    """

    def __init__(
        self,
        data: Union[Sequence, Tensor],
        targets: Union[Sequence, Tensor],
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(data) != len(targets):
            raise ValueError("Data and targets must be of equal length.")
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Return a datum and its target, after processing by transforms.

        Args:
            index (int): Index of a datum in the dataset.

        Returns:
            Tuple[Any, Any]: Datum and target pair.

        """
        datum, target = self.data[index], self.targets[index]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target


def convert_strings_to_labels(
    strings: Sequence[str], mapping: Dict[str, int], length: int
) -> Tensor:
    """
    Convert a sequence of N strings to (N, length) ndarray, with each string wrapped with <s> and </s> tokens,
    and padded wiht the <p> token.
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<p>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<s>", *tokens, "</s>"]
        for j, token in enumerate(tokens):
            labels[i, j] = mapping[token]
    return labels

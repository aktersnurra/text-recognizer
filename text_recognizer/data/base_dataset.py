"""Base PyTorch Dataset class."""
from typing import Callable, Dict, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    r"""Base Dataset class that processes data and targets through optional transfroms.

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
        transform: Callable,
        target_transform: Callable,
    ) -> None:
        super().__init__()

        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        if len(self.data) != len(self.targets):
            raise ValueError("Data and targets must be of equal length.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor]:
        """Return a datum and its target, after processing by transforms.

        Args:
            index (int): Index of a datum in the dataset.

        Returns:
            Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor]: Datum and target pair.

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
    r"""Convert a sequence of N strings to (N, length) ndarray.

    Add each string with <s> and </s> tokens, and padded wiht the <p> token.

    Args:
        strings (Sequence[str]): Sequence of strings.
        mapping (Dict[str, int]): Mapping of characters and digits to integers.
        length (int): Max lenght of all strings.

    Returns:
        Tensor: Target with emnist mapping indices.
    """
    labels = torch.ones((len(strings), length), dtype=torch.long) * mapping["<p>"]
    for i, string in enumerate(strings):
        tokens = list(string)
        tokens = ["<s>", *tokens, "<e>"]
        for j, token in enumerate(tokens):
            labels[i, j] = mapping[token]
    return labels


def split_dataset(
    dataset: BaseDataset, fraction: float, seed: int
) -> Tuple[BaseDataset, BaseDataset]:
    """Split dataset into two parts with fraction * size and (1 - fraction) * size."""
    if fraction >= 1.0:
        raise ValueError("Fraction cannot be larger greater or equal to 1.0.")
    split_1 = int(fraction * len(dataset))
    split_2 = len(dataset) - split_1
    return torch.utils.data.random_split(
        dataset, [split_1, split_2], generator=torch.Generator().manual_seed(seed)
    )

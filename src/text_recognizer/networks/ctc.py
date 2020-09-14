"""Decodes the CTC output."""
from typing import Callable, List, Optional, Tuple

from einops import rearrange
import torch
from torch import Tensor

from text_recognizer.datasets.util import EmnistMapper


def greedy_decoder(
    predictions: Tensor,
    targets: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    character_mapper: Optional[Callable] = None,
    blank_label: int = 79,
    collapse_repeated: bool = True,
) -> Tuple[List[str], List[str]]:
    """Greedy CTC decoder.

    Args:
        predictions (Tensor): Tenor of network predictions, shape [time, batch, classes].
        targets (Optional[Tensor]): Target tensor, shape is [batch, targets]. Defaults to None.
        target_lengths (Optional[Tensor]): Length of each target tensor. Defaults to None.
        character_mapper (Optional[Callable]): A emnist/character mapper for mapping integers to characters.  Defaults
            to None.
        blank_label (int): The blank character to be ignored. Defaults to 80.
        collapse_repeated (bool): Collapase consecutive predictions of the same character. Defaults to True.

    Returns:
        Tuple[List[str], List[str]]: Tuple of decoded predictions and decoded targets.

    """

    if character_mapper is None:
        character_mapper = EmnistMapper()

    predictions = rearrange(torch.argmax(predictions, dim=2), "t b -> b t")
    decoded_predictions = []
    decoded_targets = []
    for i, prediction in enumerate(predictions):
        decoded_prediction = []
        decoded_target = []
        if targets is not None and target_lengths is not None:
            for target_index in targets[i][: target_lengths[i]]:
                if target_index == blank_label:
                    continue
                decoded_target.append(character_mapper(int(target_index)))
            decoded_targets.append(decoded_target)
        for j, index in enumerate(prediction):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == prediction[j - 1]:
                    continue
                decoded_prediction.append(index.item())
        decoded_predictions.append(
            [character_mapper(int(pred_index)) for pred_index in decoded_prediction]
        )
    return decoded_predictions, decoded_targets

"""Utility functions for models."""
import Levenshtein as Lev
import torch
from torch import Tensor

from text_recognizer.networks import greedy_decoder


def accuracy(outputs: Tensor, labels: Tensor, pad_index: int = 53) -> float:
    """Computes the accuracy.

    Args:
        outputs (Tensor): The output from the network.
        labels (Tensor): Ground truth labels.
        pad_index (int): Padding index.

    Returns:
        float: The accuracy for the batch.

    """

    _, predicted = torch.max(outputs, dim=-1)

    # Mask out the pad tokens
    mask = labels != pad_index

    predicted *= mask
    labels *= mask

    acc = (predicted == labels).sum().float() / labels.shape[0]
    acc = acc.item()
    return acc


def cer(outputs: Tensor, targets: Tensor) -> float:
    """Computes the character error rate.

    Args:
        outputs (Tensor): The output from the network.
        targets (Tensor): Ground truth labels.

    Returns:
        float: The cer for the batch.

    """
    target_lengths = torch.full(
        size=(outputs.shape[1],), fill_value=targets.shape[1], dtype=torch.long,
    )
    decoded_predictions, decoded_targets = greedy_decoder(
        outputs, targets, target_lengths
    )

    lev_dist = 0

    for prediction, target in zip(decoded_predictions, decoded_targets):
        prediction = "".join(prediction)
        target = "".join(target)
        prediction, target = (
            prediction.replace(" ", ""),
            target.replace(" ", ""),
        )
        lev_dist += Lev.distance(prediction, target)
    return lev_dist / len(decoded_predictions)


def wer(outputs: Tensor, targets: Tensor) -> float:
    """Computes the Word error rate.

    Args:
        outputs (Tensor): The output from the network.
        targets (Tensor): Ground truth labels.

    Returns:
        float: The wer for the batch.

    """
    target_lengths = torch.full(
        size=(outputs.shape[1],), fill_value=targets.shape[1], dtype=torch.long,
    )
    decoded_predictions, decoded_targets = greedy_decoder(
        outputs, targets, target_lengths
    )

    lev_dist = 0

    for prediction, target in zip(decoded_predictions, decoded_targets):
        prediction = "".join(prediction)
        target = "".join(target)

        b = set(prediction.split() + target.split())
        word2char = dict(zip(b, range(len(b))))

        w1 = [chr(word2char[w]) for w in prediction.split()]
        w2 = [chr(word2char[w]) for w in target.split()]

        lev_dist += Lev.distance("".join(w1), "".join(w2))

    return lev_dist / len(decoded_predictions)

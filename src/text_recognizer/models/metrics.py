"""Utility functions for models."""

import torch


def accuracy(outputs: torch.Tensor, labels: torch.Tensro) -> float:
    """Computes the accuracy.

    Args:
        outputs (torch.Tensor): The output from the network.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: The accuracy for the batch.

    """
    _, predicted = torch.max(outputs.data, dim=1)
    acc = (predicted == labels).sum().item() / labels.shape[0]
    return acc

"""Implementations of custom loss functions."""
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss."""

    def __init__(
        self, label_smoothing: float, vocab_size: int, ignore_index: int = -100
    ) -> None:
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super().__init__()

        smoothing_value = label_smoothing / (vocab_size - 2)
        one_hot = torch.full((vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output: Tensor, targets: Tensor) -> Tensor:
        """Computes the loss.

        Args:
            output (Tensor): Predictions from the network.
            targets (Tensor): Ground truth.

        Shapes:
            outpus: Batch size x num classes
            targets: Batch size

        Returns:
            Tensor: Label smoothing loss.
        """
        model_prob = self.one_hot.repeat(targets.size(0), 1)
        model_prob.scatter_(1, targets.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((targets == self.ignore_index).unsqueeze(1), 0)
        return F.kl_div(output, model_prob, reduction="sum")

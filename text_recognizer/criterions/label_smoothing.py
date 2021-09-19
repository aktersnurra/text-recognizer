"""Implementations of custom loss functions."""
import torch
from torch import nn
from torch import Tensor


class LabelSmoothingLoss(nn.Module):
    r"""Loss functions for making networks less over confident.

    It is used to calibrate the network so that the predicted probabilities
    reflect the accuracy. The function is given by:

        L = (1 - \alpha) * y_hot + \alpha / K

    This means that some of the probability mass is transferred to the incorrect
    labels, thus not forcing the network try to put all probability mass into
    one label, and this works as a regulizer.
    """

    def __init__(
        self, ignore_index: int = -100, smoothing: float = 0.0, dim: int = -1
    ) -> None:
        super().__init__()
        assert 0.0 < smoothing <= 1.0
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """Computes the loss.

        Args:
            output (Tensor): outputictions from the network.
            target (Tensor): Ground truth.

        Shapes:
            TBC

        Returns:
            (Tensor): Label smoothing loss.
        """
        output = output.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(output)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist.masked_fill_((target == 4).unsqueeze(1), 0)
            true_dist += self.smoothing / output.size(self.dim)
        return torch.mean(torch.sum(-true_dist * output, dim=self.dim))

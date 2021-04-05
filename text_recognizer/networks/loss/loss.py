"""Implementations of custom loss functions."""
import torch
from torch import nn
from torch import Tensor

__all__ = ["LabelSmoothingCrossEntropy"]


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing loss function."""

    def __init__(
        self,
        classes: int,
        smoothing: float = 0.0,
        ignore_index: int = None,
        dim: int = -1,
    ) -> None:
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.cls = classes
        self.dim = dim

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Calculates the loss."""
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.ignore_index is not None:
                true_dist[:, self.ignore_index] = 0
                mask = torch.nonzero(target == self.ignore_index, as_tuple=False)
                if mask.dim() > 0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

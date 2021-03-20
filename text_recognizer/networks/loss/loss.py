"""Implementations of custom loss functions."""
from pytorch_metric_learning import distances, losses, miners, reducers
import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F

__all__ = ["EmbeddingLoss", "LabelSmoothingCrossEntropy"]


class EmbeddingLoss:
    """Metric loss for training encoders to produce information-rich latent embeddings."""

    def __init__(self, margin: float = 0.2, type_of_triplets: str = "semihard") -> None:
        self.distance = distances.CosineSimilarity()
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss_fn = losses.TripletMarginLoss(
            margin=margin, distance=self.distance, reducer=self.reducer
        )
        self.miner = miners.MultiSimilarityMiner(epsilon=margin, distance=self.distance)

    def __call__(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Computes the metric loss for the embeddings based on their labels.

        Args:
            embeddings (Tensor): The laten vectors encoded by the network.
            labels (Tensor): Labels of the embeddings.

        Returns:
            Tensor: The metric loss for the embeddings.

        """
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_fn(embeddings, labels, hard_pairs)
        return loss


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

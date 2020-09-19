"""Implementations of custom loss functions."""
from pytorch_metric_learning import distances, losses, miners, reducers
from torch import nn
from torch import Tensor


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

"""Character Error Rate (CER)."""
from typing import Sequence

import editdistance
import torch
from torch import Tensor
import torchmetrics


class CharacterErrorRate(torchmetrics.Metric):
    """Character error rate metric, computed using Levenshtein distance."""

    def __init__(self, ignore_tokens: Sequence[int], *args) -> None:
        super().__init__()
        self.ignore_tokens = set(ignore_tokens)
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update CER."""
        bsz = preds.shape[0]
        for index in range(bsz):
            pred = [p for p in preds[index].tolist() if p not in self.ignore_tokens]
            target = [t for t in targets[index].tolist() if t not in self.ignore_tokens]
            distance = editdistance.distance(pred, target)
            error = distance / max(len(pred), len(target))
            self.error += error
        self.total += bsz

    def compute(self) -> Tensor:
        """Compute CER."""
        return self.error / self.total

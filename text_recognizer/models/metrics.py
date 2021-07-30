"""Character Error Rate (CER)."""
from typing import Set

import attr
import editdistance
import torch
from torch import Tensor
from torchmetrics import Metric


@attr.s
class CharacterErrorRate(Metric):
    """Character error rate metric, computed using Levenshtein distance."""

    ignore_indices: Set = attr.ib(converter=set)
    error: Tensor = attr.ib(init=False)
    total: Tensor = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor) -> None:
        """Update CER."""
        bsz = preds.shape[0]
        for index in range(bsz):
            pred = [p for p in preds[index].tolist() if p not in self.ignore_indices]
            target = [
                t for t in targets[index].tolist() if t not in self.ignore_indices
            ]
            distance = editdistance.distance(pred, target)
            error = distance / max(len(pred), len(target))
            self.error += error
        self.total += bsz

    def compute(self) -> Tensor:
        """Compute CER."""
        return self.error / self.total

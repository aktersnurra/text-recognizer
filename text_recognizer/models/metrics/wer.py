"""Character Error Rate (CER)."""
from typing import Sequence

import torch
import torchmetrics


class WordErrorRate(torchmetrics.WordErrorRate):
    """Character error rate metric, allowing for tokens to be ignored."""

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_l = [
            [t for t in pred if t not in self.ignore_tokens] for pred in preds.tolist()
        ]
        targets_l = [
            [t for t in target if t not in self.ignore_tokens]
            for target in targets.tolist()
        ]
        super().update(preds_l, targets_l)

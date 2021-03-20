"""Utility functions for training neural networks."""
from typing import Dict, Optional

from loguru import logger


def log_val_metric(metrics_mean: Dict, epoch: Optional[int] = None) -> None:
    """Logging of val metrics to file/terminal."""
    log_str = "Validation metrics " + (f"at epoch {epoch} - " if epoch else " - ")
    logger.debug(log_str + " - ".join(f"{k}: {v:.4f}" for k, v in metrics_mean.items()))


class RunningAverage:
    """Maintains a running average."""

    def __init__(self) -> None:
        """Initializes the parameters."""
        self.steps = 0
        self.total = 0

    def update(self, val: float) -> None:
        """Updates the parameters."""
        self.total += val
        self.steps += 1

    def __call__(self) -> float:
        """Computes the running average."""
        return self.total / float(self.steps)

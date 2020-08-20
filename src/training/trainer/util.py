"""Utility functions for training neural networks."""


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

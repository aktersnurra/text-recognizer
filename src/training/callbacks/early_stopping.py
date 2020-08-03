"""Implements Early stopping for PyTorch model."""
from typing import Dict, Union

from loguru import logger
import numpy as np
import torch
from training.callbacks import Callback


class EarlyStopping(Callback):
    """Stops training when a monitored metric stops improving."""

    mode_dict = {
        "min": torch.lt,
        "max": torch.gt,
    }

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 3,
        mode: str = "auto",
    ) -> None:
        """Initializes the EarlyStopping callback.

        Args:
            monitor (str): Description of parameter `monitor`. Defaults to "val_loss".
            min_delta (float): Description of parameter `min_delta`. Defaults to 0.0.
            patience (int): Description of parameter `patience`. Defaults to 3.
            mode (str): Description of parameter `mode`. Defaults to "auto".

        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = torch.tensor(min_delta)
        self.mode = mode
        self.wait_count = 0
        self.stopped_epoch = 0

        if mode not in ["auto", "min", "max"]:
            logger.warning(
                f"EarlyStopping mode {mode} is unkown, fallback to auto mode."
            )

            self.mode = "auto"

        if self.mode == "auto":
            if "accuracy" in self.monitor:
                self.mode = "max"
            else:
                self.mode = "min"
            logger.debug(
                f"EarlyStopping mode set to {self.mode} for monitoring {self.monitor}."
            )

        self.torch_inf = torch.tensor(np.inf)
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        self.best_score = (
            self.torch_inf if self.monitor_op == torch.lt else -self.torch_inf
        )

    @property
    def monitor_op(self) -> float:
        """Returns the comparison method."""
        return self.mode_dict[self.mode]

    def on_fit_begin(self) -> Union[torch.lt, torch.gt]:
        """Reset the early stopping variables for reuse."""
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_score = (
            self.torch_inf if self.monitor_op == torch.lt else -self.torch_inf
        )

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        """Computes the early stop criterion."""
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_fit_end(self) -> None:
        """Logs if early stopping was used."""
        if self.stopped_epoch > 0:
            logger.info(
                f"Stopped training at epoch {self.stopped_epoch + 1} with early stopping."
            )

    def get_monitor_value(self, logs: Dict) -> Union[torch.Tensor, None]:
        """Extracts the monitor value."""
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logger.warning(
                f"Early stopping is conditioned on metric {self.monitor} which is not available. Available"
                + f"metrics are: {','.join(list(logs.keys()))}"
            )
            return None
        return torch.tensor(monitor_value)

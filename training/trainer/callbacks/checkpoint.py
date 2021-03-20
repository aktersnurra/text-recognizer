"""Callback checkpoint for training models."""
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

from loguru import logger
import numpy as np
import torch
from training.trainer.callbacks import Callback

from text_recognizer.models import Model


class Checkpoint(Callback):
    """Saving model parameters at the end of each epoch."""

    mode_dict = {
        "min": torch.lt,
        "max": torch.gt,
    }

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        monitor: str = "accuracy",
        mode: str = "auto",
        min_delta: float = 0.0,
    ) -> None:
        """Monitors a quantity that will allow us to determine the best model weights.

        Args:
            checkpoint_path (Union[str, Path]): Path to the experiment with the checkpoint.
            monitor (str): Name of the quantity to monitor. Defaults to "accuracy".
            mode (str): Description of parameter `mode`. Defaults to "auto".
            min_delta (float): Description of parameter `min_delta`. Defaults to 0.0.

        """
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.monitor = monitor
        self.mode = mode
        self.min_delta = torch.tensor(min_delta)

        if mode not in ["auto", "min", "max"]:
            logger.warning(f"Checkpoint mode {mode} is unkown, fallback to auto mode.")

            self.mode = "auto"

        if self.mode == "auto":
            if "accuracy" in self.monitor:
                self.mode = "max"
            else:
                self.mode = "min"
            logger.debug(
                f"Checkpoint mode set to {self.mode} for monitoring {self.monitor}."
            )

        torch_inf = torch.tensor(np.inf)
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def monitor_op(self) -> float:
        """Returns the comparison method."""
        return self.mode_dict[self.mode]

    def on_epoch_end(self, epoch: int, logs: Dict) -> None:
        """Saves a checkpoint for the network parameters.

        Args:
            epoch (int): The current epoch.
            logs (Dict): The log containing the monitored metrics.

        """
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            is_best = True
        else:
            is_best = False

        self.model.save_checkpoint(self.checkpoint_path, is_best, epoch, self.monitor)

    def get_monitor_value(self, logs: Dict) -> Union[float, None]:
        """Extracts the monitored value."""
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logger.warning(
                f"Checkpoint is conditioned on metric {self.monitor} which is not available. Available"
                + f" metrics are: {','.join(list(logs.keys()))}"
            )
            return None
        return monitor_value

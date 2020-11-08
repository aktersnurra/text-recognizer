"""Callbacks for learning rate schedulers."""
from typing import Callable, Dict, List, Optional, Type

from torch.optim.swa_utils import update_bn
from training.trainer.callbacks import Callback

from text_recognizer.models import Model


class LRScheduler(Callback):
    """Generic learning rate scheduler callback."""

    def __init__(self) -> None:
        super().__init__()

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and lr scheduler."""
        self.model = model
        self.lr_scheduler = self.model.lr_scheduler["lr_scheduler"]
        self.interval = self.model.lr_scheduler["interval"]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Takes a step at the end of every epoch."""
        if self.interval == "epoch":
            if "ReduceLROnPlateau" in self.lr_scheduler.__class__.__name__:
                self.lr_scheduler.step(logs["val_loss"])
            else:
                self.lr_scheduler.step()

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Takes a step at the end of every training batch."""
        if self.interval == "step":
            self.lr_scheduler.step()


class SWA(Callback):
    """Stochastic Weight Averaging callback."""

    def __init__(self) -> None:
        """Initializes the callback."""
        super().__init__()
        self.lr_scheduler = None
        self.interval = None
        self.swa_scheduler = None
        self.swa_start = None
        self.current_epoch = 1

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and lr scheduler."""
        self.model = model
        self.lr_scheduler = self.model.lr_scheduler["lr_scheduler"]
        self.interval = self.model.lr_scheduler["interval"]
        self.swa_scheduler = self.model.swa_scheduler["swa_scheduler"]
        self.swa_start = self.model.swa_scheduler["swa_start"]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Takes a step at the end of every training batch."""
        if epoch > self.swa_start:
            self.model.swa_network.update_parameters(self.model.network)
            self.swa_scheduler.step()
        elif self.interval == "epoch":
            self.lr_scheduler.step()
        self.current_epoch = epoch

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Takes a step at the end of every training batch."""
        if self.current_epoch < self.swa_start and self.interval == "step":
            self.lr_scheduler.step()

    def on_fit_end(self) -> None:
        """Update batch norm statistics for the swa model at the end of training."""
        if self.model.swa_network:
            update_bn(
                self.model.val_dataloader(),
                self.model.swa_network,
                device=self.model.device,
            )

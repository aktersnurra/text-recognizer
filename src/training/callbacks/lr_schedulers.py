"""Callbacks for learning rate schedulers."""
from typing import Callable, Dict, List, Optional, Type

from training.callbacks import Callback

from text_recognizer.models import Model


class StepLR(Callback):
    """Callback for StepLR."""

    def __init__(self) -> None:
        """Initializes the callback."""
        super().__init__()
        self.lr_scheduler = None

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and lr scheduler."""
        self.model = model
        self.lr_scheduler = self.model.lr_scheduler

    def on_epoch_end(self, epoch: int, logs: Dict = {}) -> None:
        """Takes a step at the end of every epoch."""
        self.lr_scheduler.step()


class MultiStepLR(Callback):
    """Callback for MultiStepLR."""

    def __init__(self) -> None:
        """Initializes the callback."""
        super().__init__()
        self.lr_scheduler = None

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and lr scheduler."""
        self.model = model
        self.lr_scheduler = self.model.lr_scheduler

    def on_epoch_end(self, epoch: int, logs: Dict = {}) -> None:
        """Takes a step at the end of every epoch."""
        self.lr_scheduler.step()


class ReduceLROnPlateau(Callback):
    """Callback for ReduceLROnPlateau."""

    def __init__(self) -> None:
        """Initializes the callback."""
        super().__init__()
        self.lr_scheduler = None

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and lr scheduler."""
        self.model = model
        self.lr_scheduler = self.model.lr_scheduler

    def on_epoch_end(self, epoch: int, logs: Dict = {}) -> None:
        """Takes a step at the end of every epoch."""
        val_loss = logs["val_loss"]
        self.lr_scheduler.step(val_loss)


class CyclicLR(Callback):
    """Callback for CyclicLR."""

    def __init__(self) -> None:
        """Initializes the callback."""
        super().__init__()
        self.lr_scheduler = None

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and lr scheduler."""
        self.model = model
        self.lr_scheduler = self.model.lr_scheduler

    def on_train_batch_end(self, batch: int, logs: Dict = {}) -> None:
        """Takes a step at the end of every training batch."""
        self.lr_scheduler.step()


class OneCycleLR(Callback):
    """Callback for OneCycleLR."""

    def __init__(self) -> None:
        """Initializes the callback."""
        super().__init__()
        self.lr_scheduler = None

    def set_model(self, model: Type[Model]) -> None:
        """Sets the model and lr scheduler."""
        self.model = model
        self.lr_scheduler = self.model.lr_scheduler

    def on_train_batch_end(self, batch: int, logs: Dict = {}) -> None:
        """Takes a step at the end of every training batch."""
        self.lr_scheduler.step()

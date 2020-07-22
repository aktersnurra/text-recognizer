"""Metaclass for callback functions."""

from abc import ABC
from typing import Callable, List, Type


class Callback(ABC):
    """Metaclass for callbacks used in training."""

    def on_fit_begin(self) -> None:
        """Called when fit begins."""
        pass

    def on_fit_end(self) -> None:
        """Called when fit ends."""
        pass

    def on_train_epoch_begin(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_train_epoch_end(self) -> None:
        """Called at the end of an epoch."""
        pass

    def on_val_epoch_begin(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_val_epoch_end(self) -> None:
        """Called at the end of an epoch."""
        pass

    def on_train_batch_begin(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_train_batch_end(self) -> None:
        """Called at the end of an epoch."""
        pass

    def on_val_batch_begin(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_val_batch_end(self) -> None:
        """Called at the end of an epoch."""
        pass


class CallbackList:
    """Container for abstracting away callback calls."""

    def __init__(self, callbacks: List[Callable] = None) -> None:
        """TBC."""
        self._callbacks = callbacks if callbacks is not None else []

    def append(self, callback: Type[Callback]) -> None:
        """Append new callback to callback list."""
        self.callbacks.append(callback)

    def on_fit_begin(self) -> None:
        """Called when fit begins."""
        for _ in self._callbacks:
            pass

    def on_fit_end(self) -> None:
        """Called when fit ends."""
        pass

    def on_train_epoch_begin(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_train_epoch_end(self) -> None:
        """Called at the end of an epoch."""
        pass

    def on_val_epoch_begin(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_val_epoch_end(self) -> None:
        """Called at the end of an epoch."""
        pass

    def on_train_batch_begin(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_train_batch_end(self) -> None:
        """Called at the end of an epoch."""
        pass

    def on_val_batch_begin(self) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_val_batch_end(self) -> None:
        """Called at the end of an epoch."""
        pass

"""Metaclass for callback functions."""

from enum import Enum
from typing import Callable, Dict, List, Optional, Type, Union

from loguru import logger
import numpy as np
import torch

from text_recognizer.models import Model


class ModeKeys:
    """Mode keys for CallbackList."""

    TRAIN = "train"
    VALIDATION = "validation"


class Callback:
    """Metaclass for callbacks used in training."""

    def __init__(self) -> None:
        """Initializes the Callback instance."""
        self.model = None

    def set_model(self, model: Type[Model]) -> None:
        """Set the model."""
        self.model = model

    def on_fit_begin(self) -> None:
        """Called when fit begins."""
        pass

    def on_fit_end(self) -> None:
        """Called when fit ends."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of an epoch. Only used in training mode."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch. Only used in training mode."""
        pass

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch."""
        pass

    def on_validation_batch_begin(
        self, batch: int, logs: Optional[Dict] = None
    ) -> None:
        """Called at the beginning of an epoch."""
        pass

    def on_validation_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch."""
        pass

    def on_test_begin(self) -> None:
        """Called at the beginning of test."""
        pass

    def on_test_end(self) -> None:
        """Called at the end of test."""
        pass


class CallbackList:
    """Container for abstracting away callback calls."""

    mode_keys = ModeKeys()

    def __init__(self, model: Type[Model], callbacks: List[Callback] = None) -> None:
        """Container for `Callback` instances.

        This object wraps a list of `Callback` instances and allows them all to be
        called via a single end point.

        Args:
            model (Type[Model]): A `Model` instance.
            callbacks (List[Callback]): List of `Callback` instances. Defaults to None.

        """

        self._callbacks = callbacks or []
        if model:
            self.set_model(model)

    def set_model(self, model: Type[Model]) -> None:
        """Set the model for all callbacks."""
        self.model = model
        for callback in self._callbacks:
            callback.set_model(model=self.model)

    def append(self, callback: Type[Callback]) -> None:
        """Append new callback to callback list."""
        self._callbacks.append(callback)

    def on_fit_begin(self) -> None:
        """Called when fit begins."""
        for callback in self._callbacks:
            callback.on_fit_begin()

    def on_fit_end(self) -> None:
        """Called when fit ends."""
        for callback in self._callbacks:
            callback.on_fit_end()

    def on_test_begin(self) -> None:
        """Called when test begins."""
        for callback in self._callbacks:
            callback.on_test_begin()

    def on_test_end(self) -> None:
        """Called when test ends."""
        for callback in self._callbacks:
            callback.on_test_end()

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of an epoch."""
        for callback in self._callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch."""
        for callback in self._callbacks:
            callback.on_epoch_end(epoch, logs)

    def _call_batch_hook(
        self, mode: str, hook: str, batch: int, logs: Optional[Dict] = None
    ) -> None:
        """Helper function for all batch_{begin | end} methods."""
        if hook == "begin":
            self._call_batch_begin_hook(mode, batch, logs)
        elif hook == "end":
            self._call_batch_end_hook(mode, batch, logs)
        else:
            raise ValueError(f"Unrecognized hook {hook}.")

    def _call_batch_begin_hook(
        self, mode: str, batch: int, logs: Optional[Dict] = None
    ) -> None:
        """Helper function for all `on_*_batch_begin` methods."""
        hook_name = f"on_{mode}_batch_begin"
        self._call_batch_hook_helper(hook_name, batch, logs)

    def _call_batch_end_hook(
        self, mode: str, batch: int, logs: Optional[Dict] = None
    ) -> None:
        """Helper function for all `on_*_batch_end` methods."""
        hook_name = f"on_{mode}_batch_end"
        self._call_batch_hook_helper(hook_name, batch, logs)

    def _call_batch_hook_helper(
        self, hook_name: str, batch: int, logs: Optional[Dict] = None
    ) -> None:
        """Helper function for `on_*_batch_begin` methods."""
        for callback in self._callbacks:
            hook = getattr(callback, hook_name)
            hook(batch, logs)

    def on_train_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of an epoch."""
        self._call_batch_hook(self.mode_keys.TRAIN, "begin", batch, logs)

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch."""
        self._call_batch_hook(self.mode_keys.TRAIN, "end", batch, logs)

    def on_validation_batch_begin(
        self, batch: int, logs: Optional[Dict] = None
    ) -> None:
        """Called at the beginning of an epoch."""
        self._call_batch_hook(self.mode_keys.VALIDATION, "begin", batch, logs)

    def on_validation_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch."""
        self._call_batch_hook(self.mode_keys.VALIDATION, "end", batch, logs)

    def __iter__(self) -> iter:
        """Iter function for callback list."""
        return iter(self._callbacks)

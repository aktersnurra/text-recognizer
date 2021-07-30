"""Miscellaneous neural network utility functionality."""
from functools import partial
from importlib import import_module
from typing import Any, Type

from torch import nn


def activation_function(activation: str) -> Type[nn.Module]:
    """Returns the callable activation function."""
    activation_fns = nn.ModuleDict(
        [
            ["elu", nn.ELU(inplace=True)],
            ["gelu", nn.GELU()],
            ["glu", nn.GLU()],
            ["leaky_relu", nn.LeakyReLU(negative_slope=1.0e-2, inplace=True)],
            ["none", nn.Identity()],
            ["relu", nn.ReLU(inplace=True)],
            ["selu", nn.SELU(inplace=True)],
            ["mish", nn.Mish(inplace=True)],
        ]
    )
    return activation_fns[activation.lower()]


def load_partial_fn(fn: str, **kwargs: Any) -> partial:
    """Loads partial function/class."""
    module = import_module(".".join(fn.split(".")[:-1]))
    return partial(getattr(module, fn.split(".")[-1]), **kwargs)

"""Miscellaneous neural network functionality."""
import importlib
from pathlib import Path
from typing import Dict, NamedTuple, Union, Type

from loguru import logger
import torch
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
        ]
    )
    return activation_fns[activation.lower()]

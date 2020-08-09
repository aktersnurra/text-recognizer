"""Defines the LeNet network."""
from typing import Callable, Dict, Optional, Tuple

from einops.layers.torch import Rearrange
import torch
from torch import nn


class LeNet(nn.Module):
    """LeNet network."""

    def __init__(
        self,
        channels: Tuple[int, ...] = (1, 32, 64),
        kernel_sizes: Tuple[int, ...] = (3, 3, 2),
        hidden_size: Tuple[int, ...] = (9216, 128),
        dropout_rate: float = 0.2,
        output_size: int = 10,
        activation_fn: Optional[Callable] = None,
        activation_fn_args: Optional[Dict] = None,
    ) -> None:
        """The LeNet network.

        Args:
            channels (Tuple[int, ...]): Channels in the convolutional layers. Defaults to (1, 32, 64).
            kernel_sizes (Tuple[int, ...]): Kernel sizes in the convolutional layers. Defaults to (3, 3, 2).
            hidden_size (Tuple[int, ...]): Size of the flattend output form the convolutional layers.
                Defaults to (9216, 128).
            dropout_rate (float): The dropout rate. Defaults to 0.2.
            output_size (int): Number of classes. Defaults to 10.
            activation_fn (Optional[Callable]): The non-linear activation function. Defaults to
                nn.ReLU(inplace).
            activation_fn_args (Optional[Dict]): The arguments for the activation function. Defaults to None.

        """
        super().__init__()

        if activation_fn is not None:
            activation_fn_args = activation_fn_args or {}
            activation_fn = getattr(nn, activation_fn)(**activation_fn_args)
        else:
            activation_fn = nn.ReLU(inplace=True)

        self.layers = [
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=kernel_sizes[0],
            ),
            activation_fn,
            nn.Conv2d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=kernel_sizes[1],
            ),
            activation_fn,
            nn.MaxPool2d(kernel_sizes[2]),
            nn.Dropout(p=dropout_rate),
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1]),
            activation_fn,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=hidden_size[1], out_features=output_size),
        ]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The feedforward."""
        # If batch dimenstion is missing, it needs to be added.
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.layers(x)

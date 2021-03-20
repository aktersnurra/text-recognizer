"""Implementation of the LeNet network."""
from typing import Callable, Dict, Optional, Tuple

from einops.layers.torch import Rearrange
import torch
from torch import nn

from text_recognizer.networks.util import activation_function


class LeNet(nn.Module):
    """LeNet network for character prediction."""

    def __init__(
        self,
        channels: Tuple[int, ...] = (1, 32, 64),
        kernel_sizes: Tuple[int, ...] = (3, 3, 2),
        hidden_size: Tuple[int, ...] = (9216, 128),
        dropout_rate: float = 0.2,
        num_classes: int = 10,
        activation_fn: Optional[str] = "relu",
    ) -> None:
        """Initialization of the LeNet network.

        Args:
            channels (Tuple[int, ...]): Channels in the convolutional layers. Defaults to (1, 32, 64).
            kernel_sizes (Tuple[int, ...]): Kernel sizes in the convolutional layers. Defaults to (3, 3, 2).
            hidden_size (Tuple[int, ...]): Size of the flattend output form the convolutional layers.
                Defaults to (9216, 128).
            dropout_rate (float): The dropout rate. Defaults to 0.2.
            num_classes (int): Number of classes. Defaults to 10.
            activation_fn (Optional[str]): The name of non-linear activation function. Defaults to relu.

        """
        super().__init__()

        activation_fn = activation_function(activation_fn)

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
            nn.Linear(in_features=hidden_size[1], out_features=num_classes),
        ]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The feedforward pass."""
        # If batch dimenstion is missing, it needs to be added.
        if len(x.shape) < 4:
            x = x[(None,) * (4 - len(x.shape))]
        return self.layers(x)

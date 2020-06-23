"""Defines the LeNet network."""
from typing import Callable, Optional, Tuple

import torch
from torch import nn


class Flatten(nn.Module):
    """Flattens a tensor."""

    def forward(self, x: int) -> torch.Tensor:
        """Flattens a tensor for input to a nn.Linear layer."""
        return torch.flatten(x, start_dim=1)


class LeNet(nn.Module):
    """LeNet network."""

    def __init__(
        self,
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
        hidden_size: Tuple[int, ...],
        dropout_rate: float,
        output_size: int,
        activation_fn: Optional[Callable] = None,
    ) -> None:
        """The LeNet network.

        Args:
            channels (Tuple[int, ...]): Channels in the convolutional layers.
            kernel_sizes (Tuple[int, ...]): Kernel sizes in the convolutional layers.
            hidden_size (Tuple[int, ...]): Size of the flattend output form the convolutional layers.
            dropout_rate (float): The dropout rate.
            output_size (int): Number of classes.
            activation_fn (Optional[Callable]): The non-linear activation function. Defaults to
                nn.ReLU(inplace).

        """
        super().__init__()

        if activation_fn is None:
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
            Flatten(),
            nn.Linear(in_features=hidden_size[0], out_features=hidden_size[1]),
            activation_fn,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=hidden_size[1], out_features=output_size),
        ]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The feedforward."""
        return self.layers(x)


# def test():
#     x = torch.randn([1, 1, 28, 28])
#     channels = [1, 32, 64]
#     kernel_sizes = [3, 3, 2]
#     hidden_size = [9216, 128]
#     output_size = 10
#     dropout_rate = 0.2
#     activation_fn = nn.ReLU()
#     net = LeNet(
#         channels=channels,
#         kernel_sizes=kernel_sizes,
#         dropout_rate=dropout_rate,
#         hidden_size=hidden_size,
#         output_size=output_size,
#         activation_fn=activation_fn,
#     )
#     from torchsummary import summary
#
#     summary(net, (1, 28, 28), device="cpu")
#     out = net(x)

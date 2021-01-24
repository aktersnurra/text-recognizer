"""Implementation of a simple backbone cnn network."""
from typing import Callable, Dict, Optional, Tuple

from einops.layers.torch import Rearrange
import torch
from torch import nn

from text_recognizer.networks.util import activation_function


class CNN(nn.Module):
    """LeNet network for character prediction."""

    def __init__(
        self,
        channels: Tuple[int, ...] = (1, 32, 64, 128),
        kernel_sizes: Tuple[int, ...] = (4, 4, 4),
        strides: Tuple[int, ...] = (2, 2, 2),
        max_pool_kernel: int = 2,
        dropout_rate: float = 0.2,
        activation: Optional[str] = "relu",
    ) -> None:
        """Initialization of the LeNet network.

        Args:
            channels (Tuple[int, ...]): Channels in the convolutional layers. Defaults to (1, 32, 64).
            kernel_sizes (Tuple[int, ...]): Kernel sizes in the convolutional layers. Defaults to (3, 3, 2).
            strides (Tuple[int, ...]): Stride length of the convolutional filter. Defaults to (2, 2, 2).
            max_pool_kernel (int): 2D max pooling kernel. Defaults to 2.
            dropout_rate (float): The dropout rate. Defaults to 0.2.
            activation (Optional[str]): The name of non-linear activation function. Defaults to relu.

        Raises:
            RuntimeError: if the number of hyperparameters does not match in length.

        """
        super().__init__()

        if len(channels) - 1 != len(kernel_sizes) and len(kernel_sizes) != len(strides):
            raise RuntimeError("The number of the hyperparameters does not match.")

        self.cnn = self._build_network(
            channels, kernel_sizes, strides, max_pool_kernel, dropout_rate, activation,
        )

    def _build_network(
        self,
        channels: Tuple[int, ...],
        kernel_sizes: Tuple[int, ...],
        strides: Tuple[int, ...],
        max_pool_kernel: int,
        dropout_rate: float,
        activation: str,
    ) -> nn.Sequential:
        # Load activation function.
        activation_fn = activation_function(activation)

        channels = list(channels)
        in_channels = channels.pop(0)
        configuration = zip(channels, kernel_sizes, strides)

        modules = nn.ModuleList([])

        for i, (out_channels, kernel_size, stride) in enumerate(configuration):
            # Add max pool to reduce output size.
            if i == len(channels) // 2:
                modules.append(nn.MaxPool2d(max_pool_kernel))
            if i == 0:
                modules.append(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size, stride=stride, padding=1
                    )
                )
            else:
                modules.append(
                    nn.Sequential(
                        activation_fn,
                        nn.BatchNorm2d(in_channels),
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=1,
                        ),
                    )
                )

            if dropout_rate:
                modules.append(nn.Dropout2d(p=dropout_rate))

            in_channels = out_channels

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The feedforward pass."""
        # If batch dimenstion is missing, it needs to be added.
        if len(x.shape) < 4:
            x = x[(None,) * (4 - len(x.shape))]
        return self.cnn(x)

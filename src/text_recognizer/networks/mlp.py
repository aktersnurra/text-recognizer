"""Defines the MLP network."""
from typing import Callable, Dict, List, Optional, Union

from einops.layers.torch import Rearrange
import torch
from torch import nn

from text_recognizer.networks.misc import activation_function


class MLP(nn.Module):
    """Multi layered perceptron network."""

    def __init__(
        self,
        input_size: int = 784,
        num_classes: int = 10,
        hidden_size: Union[int, List] = 128,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        activation_fn: str = "relu",
    ) -> None:
        """Initialization of the MLP network.

        Args:
            input_size (int): The input shape of the network. Defaults to 784.
            num_classes (int): Number of classes in the dataset. Defaults to 10.
            hidden_size (Union[int, List]): The number of `neurons` in each hidden layer. Defaults to 128.
            num_layers (int): The number of hidden layers. Defaults to 3.
            dropout_rate (float): The dropout rate at each layer. Defaults to 0.2.
            activation_fn (str): Name of the activation function in the hidden layers. Defaults to
                relu.

        """
        super().__init__()

        activation_fn = activation_function(activation_fn)

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * num_layers

        self.layers = [
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(in_features=input_size, out_features=hidden_size[0]),
            activation_fn,
        ]

        for i in range(num_layers - 1):
            self.layers += [
                nn.Linear(in_features=hidden_size[i], out_features=hidden_size[i + 1]),
                activation_fn,
            ]

            if dropout_rate:
                self.layers.append(nn.Dropout(p=dropout_rate))

        self.layers.append(
            nn.Linear(in_features=hidden_size[-1], out_features=num_classes)
        )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The feedforward pass."""
        # If batch dimenstion is missing, it needs to be added.
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.layers(x)

    @property
    def __name__(self) -> str:
        """Returns the name of the network."""
        return "mlp"

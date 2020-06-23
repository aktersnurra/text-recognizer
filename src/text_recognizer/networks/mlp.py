"""Defines the MLP network."""
from typing import Callable, Optional

import torch
from torch import nn


class MLP(nn.Module):
    """Multi layered perceptron network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        activation_fn: Optional[Callable] = None,
    ) -> None:
        """Initialization of the MLP network.

        Args:
            input_size (int): The input shape of the network.
            output_size (int): Number of classes in the dataset.
            hidden_size (int): The number of `neurons` in each hidden layer.
            num_layers (int): The number of hidden layers.
            dropout_rate (float): The dropout rate at each layer.
            activation_fn (Optional[Callable]): The activation function in the hidden layers, (default:
                nn.ReLU()).

        """
        super().__init__()

        if activation_fn is None:
            activation_fn = nn.ReLU(inplace=True)

        self.layers = [
            nn.Linear(in_features=input_size, out_features=hidden_size),
            activation_fn,
        ]

        for _ in range(num_layers):
            self.layers += [
                nn.Linear(in_features=hidden_size, out_features=hidden_size),
                activation_fn,
            ]

            if dropout_rate:
                self.layers.append(nn.Dropout(p=dropout_rate))

        self.layers.append(nn.Linear(in_features=hidden_size, out_features=output_size))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The feedforward."""
        x = torch.flatten(x, start_dim=1)
        return self.layers(x)


# def test():
#     x = torch.randn([1, 28, 28])
#     input_size = torch.flatten(x).shape[0]
#     output_size = 10
#     hidden_size = 128
#     num_layers = 5
#     dropout_rate = 0.25
#     activation_fn = nn.GELU()
#     net = MLP(
#         input_size=input_size,
#         output_size=output_size,
#         hidden_size=hidden_size,
#         num_layers=num_layers,
#         dropout_rate=dropout_rate,
#         activation_fn=activation_fn,
#     )
#     from torchsummary import summary
#
#     summary(net, (1, 28, 28), device="cpu")
#
#     out = net(x)

"""Spatial Transformer Network."""

from einops.layers.torch import Rearrange
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class SpatialTransformerNetwork(nn.Module):
    """A network with differentiable attention.

    Network that learns how to perform spatial transformations on the input image in order to enhance the
    geometric invariance of the model.

    # TODO: add arguements to make it more general.

    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize the identity transformation and its weights and biases.
        linear = nn.Linear(32, 3 * 2)
        linear.weight.data.zero_()
        linear.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            Rearrange("b c h w -> b (c h w)", h=3, w=3),
            nn.Linear(in_features=10 * 3 * 3, out_features=32),
            nn.ReLU(inplace=True),
            linear,
            Rearrange("b (row col) -> b row col", row=2, col=3),
        )

    def forward(self, x: Tensor) -> Tensor:
        """The spatial transformation."""
        grid = F.affine_grid(self.theta(x), x.shape)
        return F.grid_sample(x, grid, align_corners=False)

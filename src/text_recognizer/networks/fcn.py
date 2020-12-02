"""Fully Convolutional Network (FCN) with dilated kernels for global context."""
from typing import List, Tuple, Type
import torch
from torch import nn
from torch import Tensor


from text_recognizer.networks.util import activation_function


class _DilatedBlock(nn.Module):
    def __init__(
        self,
        channels: List[int],
        kernel_sizes: List[int],
        dilations: List[int],
        paddings: List[int],
        activation_fn: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.dilation_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=kernel_sizes[0],
                stride=1,
                dilation=dilations[0],
                padding=paddings[0],
            ),
            nn.Conv2d(
                in_channels=channels[1],
                out_channels=channels[1] // 2,
                kernel_size=kernel_sizes[1],
                stride=1,
                dilation=dilations[1],
                padding=paddings[1],
            ),
        )
        self.activation_fn = activation_fn

        self.conv = nn.Conv2d(
            in_channels=channels[0],
            out_channels=channels[1] // 2,
            kernel_size=1,
            dilation=1,
            stride=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv(x)
        x = self.dilation_conv(x)
        x = torch.cat((x, residual), dim=1)
        return self.activation_fn(x)


class FCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        out_channels: int,
        kernel_size: int,
        dilations: Tuple[int] = (3, 7),
        paddings: Tuple[int] = (9, 21),
        num_blocks: int = 14,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.kernel_sizes = [kernel_size] * num_blocks
        self.channels = [in_channels] + [base_channels] * (num_blocks - 1)
        self.out_channels = out_channels
        self.dilations = [dilations[0]] * (num_blocks // 2) + [dilations[1]] * (
            num_blocks // 2
        )
        self.paddings = [paddings[0]] * (num_blocks // 2) + [paddings[1]] * (
            num_blocks // 2
        )
        self.activation_fn = activation_function(activation)
        self.fcn = self._configure_fcn()

    def _configure_fcn(self) -> nn.Sequential:
        layers = []
        for i in range(0, len(self.channels), 2):
            layers.append(
                _DilatedBlock(
                    self.channels[i : i + 2],
                    self.kernel_sizes[i : i + 2],
                    self.dilations[i : i + 2],
                    self.paddings[i : i + 2],
                    self.activation_fn,
                )
            )
        layers.append(
            nn.Conv2d(self.channels[-1], self.out_channels, kernel_size=1, stride=1)
        )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.fcn(x)

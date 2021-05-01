"""Efficient net b0 implementation."""
import torch
from torch import nn
from torch import Tensor


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, reduce_dim: int) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [C, H, W] -> [C, 1, 1]
            nn.Conv2d(in_channels=in_channels, out_channels=reduce_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=reduce_dim, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)


class InvertedResidulaBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        expand_ratio: float,
        reduction: int = 4,
        survival_prob: float = 0.8,
    ) -> None:
        super().__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduce_dim = in_channels // reduction

        if self.expand:
            self.expand_conv = ConvNorm(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
            )

        self.conv = nn.Sequential(
            ConvNorm(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim
            ),
            SqueezeExcite(hidden_dim, reduce_dim),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def stochastic_depth(self, x: Tensor) -> Tensor:
        if not self.training:
            return x

        binary_tensor = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        )
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, x: Tensor) -> Tensor:
        out = self.expand_conv(x) if self.expand else x
        if self.use_residual:
            return self.stochastic_depth(self.conv(out)) + x
        return self.conv(out)


class EfficientNet(nn.Module):
    """Efficient net b0 backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.base_model = [
            # expand_ratio, channels, repeats, stride, kernel_size
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        self.backbone = self._build_b0()

    def _build_b0(self) -> nn.Sequential:
        in_channels = 32
        layers = [ConvNorm(1, in_channels, 3, stride=2, padding=1)]

        for expand_ratio, out_channels, repeats, stride, kernel_size in self.base_model:
            for i in range(repeats):
                layers.append(
                    InvertedResidulaBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if i == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
                in_channels = out_channels
        layers.append(ConvNorm(in_channels, 256, kernel_size=1, stride=1, padding=0))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

"""Time-Depth Separable Convolutions.

References:
    https://arxiv.org/abs/1904.02619
    https://arxiv.org/pdf/2010.01003.pdf

Code stolen from:
    https://github.com/facebookresearch/gtn_applications


"""
from typing import List, Tuple

from einops import rearrange
import gtn
import numpy as np
import torch
from torch import nn
from torch import Tensor


class TDSBlock2d(nn.Module):
    """Internal block of a 2D TDSC network."""

    def __init__(
        self,
        in_channels: int,
        img_depth: int,
        kernel_size: Tuple[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.img_depth = img_depth
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.fc_dim = in_channels * img_depth

        # Network placeholders.
        self.conv = None
        self.mlp = None
        self.instance_norm = None

        self._build_block()

    def _build_block(self) -> None:
        # Convolutional block.
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=(1, self.kernel_size[0], self.kernel_size[1]),
                padding=(0, self.kernel_size[0] // 2, self.kernel_size[1] // 2),
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
        )

        # MLP block.
        self.mlp = nn.Sequential(
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_dim, self.fc_dim),
            nn.Dropout(self.dropout_rate),
        )

        # Instance norm.
        self.instance_norm = nn.ModuleList(
            [
                nn.InstanceNorm2d(self.fc_dim, affine=True),
                nn.InstanceNorm2d(self.fc_dim, affine=True),
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Shape:
            - x: :math: `(B, CD, H, W)`

        Returns:
            Tensor: Output tensor.

        """
        B, CD, H, W = x.shape
        C, D = self.in_channels, self.img_depth
        residual = x
        x = rearrange(x, "b (c d) h w -> b c d h w", c=C, d=D)
        x = self.conv(x)
        x = rearrange(x, "b c d h w -> b (c d) h w")
        x += residual

        x = self.instance_norm[0](x)

        x = self.mlp(x.transpose(1, 3)).transpose(1, 3) + x
        x + self.instance_norm[1](x)

        # Output shape: [B, CD, H, W]
        return x


class TDS2d(nn.Module):
    """TDS Netowrk.

    Structure is the following:
        Downsample layer -> TDS2d group -> ... -> Linear output layer


    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        tds_groups: Tuple[int],
        kernel_size: Tuple[int],
        dropout_rate: float,
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.tds_groups = tds_groups
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.tds = None
        self.fc = None

    def _build_network(self) -> None:

        modules = []
        stride_h = np.prod([grp["stride"][0] for grp in self.tds_groups])
        if self.input_dim % stride_h:
            raise RuntimeError(
                f"Image height not divisible by total stride {stride_h}."
            )

        for tds_group in self.tds_groups:
            # Add downsample layer.
            out_channels = self.depth * tds_group["channels"]
            modules.extend(
                [
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        kernel_size=self.kernel_size,
                        padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2),
                        stride=tds_group["stride"],
                    ),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout_rate),
                    nn.InstanceNorm2d(out_channels, affine=True),
                ]
            )

            for _ in range(tds_group["num_blocks"]):
                modules.append(
                    TDSBlock2d(
                        tds_group["channels"],
                        self.depth,
                        self.kernel_size,
                        self.dropout_rate,
                    )
                )

            self.in_channels = out_channels

        self.tds = nn.Sequential(*modules)
        self.fc = nn.Linear(
            self.in_channels * self.input_dim // stride_h, self.output_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Shape:
            - x: :math: `(B, H, W)`

        Returns:
            Tensor: Output tensor.

        """
        B, H, W = x.shape
        x = rearrange(
            x, "b (h1 h2) w -> b h1 h2 w", h1=self.in_channels, h2=H // self.in_channels
        )
        x = self.tds(x)

        # x shape: [B, C, H, W]
        x = rearrange(x, "b c h w -> b w (c h)")

        return self.fc(x)

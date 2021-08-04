"""Convolutional attention block."""
import attr
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from text_recognizer.networks.vqvae.norm import Normalize


@attr.s
class Attention(nn.Module):
    """Convolutional attention."""

    in_channels: int = attr.ib()
    q: nn.Conv2d = attr.ib(init=False)
    k: nn.Conv2d = attr.ib(init=False)
    v: nn.Conv2d = attr.ib(init=False)
    proj: nn.Conv2d = attr.ib(init=False)
    norm: Normalize = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        """Post init configuration."""
        super().__init__()
        self.q = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.norm = Normalize(num_channels=self.in_channels)
        self.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applies attention to feature maps."""
        residual = x
        x = self.norm(x)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Attention
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k = k.reshape(B, C, H * W)  # [B, C, HW]
        energy = torch.bmm(q, k) * (C ** -0.5)
        attention = F.softmax(energy, dim=2)

        # Compute attention to which values
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        out = torch.bmm(v, attention)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        return out + residual

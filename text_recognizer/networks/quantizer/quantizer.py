"""Implementation of a Vector Quantized Variational AutoEncoder.

Reference:
https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
"""
from typing import Tuple, Type

import attr
from einops import rearrange
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


@attr.s(eq=False)
class VectorQuantizer(nn.Module):
    """Vector quantizer."""

    input_dim: int = attr.ib()
    codebook: Type[nn.Module] = attr.ib()
    commitment: float = attr.ib(default=1.0)
    project_in: nn.Linear = attr.ib(default=None, init=False)
    project_out: nn.Linear = attr.ib(default=None, init=False)

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    def __attrs_post_init__(self) -> None:
        require_projection = self.codebook.dim != self.input_dim
        self.project_in = (
            nn.Linear(self.input_dim, self.codebook.dim)
            if require_projection
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(self.codebook.dim, self.input_dim)
            if require_projection
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Quantizes latent vectors."""
        H, W = x.shape[-2:]
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.project_in(x)

        quantized, indices = self.codebook(x)

        if self.training:
            commitment_loss = F.mse_loss(quantized.detach(), x) * self.commitment
            quantized = x + (quantized - x).detach()
        else:
            commitment_loss = torch.tensor([0.0]).type_as(x)

        quantized = self.project_out(quantized)
        quantized = rearrange(quantized, "b (h w) d -> b d h w", h=H, w=W)

        return quantized, indices, commitment_loss

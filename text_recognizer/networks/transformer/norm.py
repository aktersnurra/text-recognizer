"""Normalization layers for transfromers.

Copied from lucidrains:
    https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

"""
from typing import Optional, Type

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Applies normalization."""
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class PreNorm(nn.Module):
    """Applies layer normalization then function."""

    def __init__(
        self,
        normalized_shape: int,
        fn: Type[nn.Module],
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        self.fn = fn
        self.norm_context = (
            nn.LayerNorm(context_dim) if context_dim is not None else None
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Applies pre norm."""
        x = self.norm(x)
        if self.norm_context is not None:
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)

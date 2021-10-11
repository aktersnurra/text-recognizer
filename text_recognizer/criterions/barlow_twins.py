"""Barlow twins loss function."""

import torch
from torch import nn, Tensor


def off_diagonal(x: Tensor) -> Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsLoss(nn.Module):
    def __init__(self, dim: int, lambda_: float) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False)
        self.lambda_ = lambda_

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Calculates the Barlow Twin loss."""
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambda_ * off_diag

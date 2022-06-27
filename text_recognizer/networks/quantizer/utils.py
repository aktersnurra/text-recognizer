"""Helper functions for quantization."""
from typing import Tuple

import torch
from torch import einsum, Tensor
import torch.nn.functional as F


def sample_vectors(samples: Tensor, num: int) -> Tensor:
    """Subsamples a set of vectors."""
    B, device = samples.shape[0], samples.device
    if B >= num:
        indices = torch.randperm(B, device=device)[:num]
    else:
        indices = torch.randint(0, B, (num,), device=device)[:num]
    return samples[indices]


def norm(t: Tensor) -> Tensor:
    """Applies L2-normalization."""
    return F.normalize(t, p=2, dim=-1)


def ema_inplace(moving_avg: Tensor, new: Tensor, decay: float) -> None:
    """Applies exponential moving average."""
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def log(t: Tensor, eps: float = 1e-20) -> Tensor:
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t: Tensor) -> Tensor:
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t: Tensor, temperature: float = 1.0, dim: int = -1) -> Tensor:
    if temperature == 0:
        return t.argmax(dim=dim)
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=dim)


def orthgonal_loss_fn(t: Tensor) -> Tensor:
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = einsum("i d, j d -> i j", normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)

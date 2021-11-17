"""Helper functions for quantization."""
from typing import Tuple

import torch
from torch import Tensor
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

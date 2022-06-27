"""K-means clustering for embeddings."""
from typing import Tuple

from einops import repeat
import torch
from torch import Tensor

from text_recognizer.networks.quantizer.utils import norm, sample_vectors


def kmeans(
    samples: Tensor, num_clusters: int, num_iters: int = 10
) -> Tuple[Tensor, Tensor]:
    """Compute k-means clusters."""
    D = samples.shape[-1]

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        dists = samples @ means.t()
        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, D).type_as(samples)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=D), samples)
        new_means /= bins_min_clamped[..., None]
        new_means = norm(new_means)
        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

"""Codebook module."""
from typing import Tuple

from einops import rearrange
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from text_recognizer.networks.quantizer.kmeans import kmeans
from text_recognizer.networks.quantizer.utils import (
    ema_inplace,
    norm,
    sample_vectors,
    gumbel_sample,
)


class CosineSimilarityCodebook(nn.Module):
    """Cosine similarity codebook."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        eps: float = 1.0e-5,
        threshold_dead: int = 2,
        temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.decay = decay
        self.eps = eps
        self.threshold_dead = threshold_dead
        self.temperature = temperature

        if not self.kmeans_init:
            embeddings = norm(torch.randn(self.codebook_size, self.dim))
        else:
            embeddings = torch.zeros(self.codebook_size, self.dim)
        self.register_buffer("initalized", Tensor([not self.kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size))
        self.register_buffer("embeddings", embeddings)

    def _initalize_embedding(self, data: Tensor) -> None:
        embeddings, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embeddings.data.copy_(embeddings)
        self.cluster_size.data.copy_(cluster_size)
        self.initalized.data.copy_(Tensor([True]))

    def _replace(self, samples: Tensor, mask: Tensor) -> None:
        samples = norm(samples)
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embeddings,
        )
        self.embeddings.data.copy_(modified_codebook)

    def _replace_dead_codes(self, batch_samples: Tensor) -> None:
        if self.threshold_dead == 0:
            return
        dead_codes = self.cluster_size < self.threshold_dead
        if not torch.any(dead_codes):
            return
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self._replace(batch_samples, mask=dead_codes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantizes tensor."""
        shape = x.shape
        flatten = rearrange(x, "... d -> (...) d")
        flatten = norm(flatten)

        if not self.initalized:
            self._initalize_embedding(flatten)

        embeddings = norm(self.embeddings)
        dist = flatten @ embeddings.t()
        indices = gumbel_sample(dist, dim=-1, temperature=self.temperature)
        one_hot = F.one_hot(indices, self.codebook_size).type_as(x)
        indices = indices.view(*shape[:-1])

        quantized = F.embedding(indices, self.embeddings)

        if self.training:
            bins = one_hot.sum(0)
            ema_inplace(self.cluster_size, bins, self.decay)
            zero_mask = bins == 0
            bins = bins.masked_fill(zero_mask, 1.0)

            embed_sum = flatten.t() @ one_hot
            embed_norm = (embed_sum / bins.unsqueeze(0)).t()
            embed_norm = norm(embed_norm)
            embed_norm = torch.where(zero_mask[..., None], embeddings, embed_norm)
            ema_inplace(self.embeddings, embed_norm, self.decay)
            self._replace_dead_codes(x)

        return quantized, indices
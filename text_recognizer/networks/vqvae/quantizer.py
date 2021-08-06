"""Implementation of a Vector Quantized Variational AutoEncoder.

Reference:
https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
"""
from einops import rearrange
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class EmbeddingEMA(nn.Module):
    """Embedding for Exponential Moving Average (EMA)."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        weight = torch.zeros(num_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(weight, nonlinearity="linear")
        self.register_buffer("weight", weight)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("weight_avg", weight.clone())


class VectorQuantizer(nn.Module):
    """The codebook that contains quantized vectors."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, decay: float = 0.99
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.embedding = EmbeddingEMA(self.num_embeddings, self.embedding_dim)

    def discretization_bottleneck(self, latent: Tensor) -> Tensor:
        """Computes the code nearest to the latent representation.

        First we compute the posterior categorical distribution, and then map
        the latent representation to the nearest element of the embedding.

        Args:
            latent (Tensor): The latent representation.

        Shape:
            - latent :math:`(B x H x W, D)`

        Returns:
            Tensor: The quantized embedding vector.

        """
        # Store latent shape.
        b, h, w, d = latent.shape

        # Flatten the latent representation to 2D.
        latent = rearrange(latent, "b h w d -> (b h w) d")

        # Compute the L2 distance between the latents and the embeddings.
        l2_distance = (
            torch.sum(latent ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * latent @ self.embedding.weight.t()
        )  # [BHW x K]

        # Find the embedding k nearest to each latent.
        encoding_indices = torch.argmin(l2_distance, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings, aka discrete bottleneck.
        one_hot_encoding = torch.zeros(
            encoding_indices.shape[0], self.num_embeddings, device=latent.device
        )
        one_hot_encoding.scatter_(1, encoding_indices, 1)  # [BHW x K]

        # Embedding quantization.
        quantized_latent = one_hot_encoding @ self.embedding.weight  # [BHW, D]
        quantized_latent = rearrange(
            quantized_latent, "(b h w) d -> b h w d", b=b, h=h, w=w
        )
        if self.training:
            self.compute_ema(one_hot_encoding=one_hot_encoding, latent=latent)

        return quantized_latent

    def compute_ema(self, one_hot_encoding: Tensor, latent: Tensor) -> None:
        """Computes the EMA update to the codebook."""
        batch_cluster_size = one_hot_encoding.sum(axis=0)
        batch_embedding_avg = (latent.t() @ one_hot_encoding).t()
        self.embedding.cluster_size.data.mul_(self.decay).add_(
            batch_cluster_size, alpha=1 - self.decay
        )
        self.embedding.weight_avg.data.mul_(self.decay).add_(
            batch_embedding_avg, alpha=1 - self.decay
        )
        new_embedding = self.embedding.weight_avg / (
            self.embedding.cluster_size + 1.0e-5
        ).unsqueeze(1)
        self.embedding.weight.data.copy_(new_embedding)

    def vq_loss(self, latent: Tensor, quantized_latent: Tensor) -> Tensor:
        """Vector Quantization loss.

        The vector quantization algorithm allows us to create a codebook. The VQ
        algorithm works by moving the embedding vectors towards the encoder outputs.

        The embedding loss moves the embedding vector towards the encoder outputs. The
        .detach() works as the stop gradient (sg) described in the paper.

        Because the volume of the embedding space is dimensionless, it can arbitarily
        grow if the embeddings are not trained as fast as the encoder parameters. To
        mitigate this, a commitment loss is added in the second term which makes sure
        that the encoder commits to an embedding and that its output does not grow.

        Args:
            latent (Tensor): The encoder output.
            quantized_latent (Tensor): The quantized latent.

        Returns:
            Tensor: The combinded VQ loss.

        """
        commitment_loss = F.mse_loss(quantized_latent.detach(), latent)
        # embedding_loss = F.mse_loss(quantized_latent, latent.detach())
        # return embedding_loss + self.beta * commitment_loss
        return commitment_loss

    def forward(self, latent: Tensor) -> Tensor:
        """Forward pass that returns the quantized vector and the vq loss."""
        # Rearrange latent representation s.t. the hidden dim is at the end.
        latent = rearrange(latent, "b d h w -> b h w d")

        # Maps latent to the nearest code in the codebook.
        quantized_latent = self.discretization_bottleneck(latent)

        loss = self.vq_loss(latent, quantized_latent)

        # Add residue to the quantized latent.
        quantized_latent = latent + (quantized_latent - latent).detach()

        # Rearrange the quantized shape back to the original shape.
        quantized_latent = rearrange(quantized_latent, "b h w d -> b d h w")

        return quantized_latent, loss

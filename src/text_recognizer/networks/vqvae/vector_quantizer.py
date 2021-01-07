"""Implementation of a Vector Quantized Variational AutoEncoder.

Reference:
https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py

"""

from einops import rearrange
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """The codebook that contains quantized vectors."""

    def __init__(
        self, num_embeddings: int, embedding_dim: int, beta: float = 0.25
    ) -> None:
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)

        # Initialize the codebook.
        self.embedding.weight.uniform_(-1 / self.K, 1 / self.K)

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
            encoding_indices.shape[0], self.K, device=latent.device
        )
        one_hot_encoding.scatter_(1, encoding_indices, 1)  # [BHW x K]

        # Embedding quantization.
        quantized_latent = one_hot_encoding @ self.embedding.weight  # [BHW, D]
        quantized_latent = rearrange(
            quantized_latent, "(b h w) d -> b h w d", b=b, h=h, w=w
        )

        return quantized_latent

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
        embedding_loss = F.mse_loss(quantized_latent, latent.detach())
        commitment_loss = F.mse_loss(quantized_latent.detach(), latent)
        return embedding_loss + self.beta * commitment_loss

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

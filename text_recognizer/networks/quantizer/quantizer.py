from typing import Optional, Tuple, Type

from einops import rearrange
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from text_recognizer.networks.quantizer.utils import orthgonal_loss_fn


class VectorQuantizer(nn.Module):
    """Vector quantizer."""

    def __init__(
        self,
        input_dim: int,
        codebook: Type[nn.Module],
        commitment: float = 1.0,
        ort_reg_weight: float = 0,
        ort_reg_max_codes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.codebook = codebook
        self.commitment = commitment
        self.ort_reg_weight = ort_reg_weight
        self.ort_reg_max_codes = ort_reg_max_codes
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
        device = x.device
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.project_in(x)

        quantized, indices = self.codebook(x)

        if self.training:
            loss = F.mse_loss(quantized.detach(), x) * self.commitment
            quantized = x + (quantized - x).detach()
            if self.ort_reg_weight > 0:
                codebook = self.codebook.embeddings
                num_codes = codebook.shape[0]
                if num_codes > self.ort_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=device)[
                        : self.ort_reg_max_codes
                    ]
                    codebook = codebook[rand_ids]
                orthgonal_loss = orthgonal_loss_fn(codebook)
                loss += self.ort_reg_weight * orthgonal_loss
        else:
            loss = torch.tensor([0.0]).type_as(x)

        quantized = self.project_out(quantized)
        quantized = rearrange(quantized, "b (h w) d -> b d h w", h=H, w=W)

        return quantized, indices, loss

"""Nyströmer encoder.

Efficient attention module that reduces the complexity of the attention module from
O(n**2) to O(n). The attention matrix is assumed low rank and thus the information 
can be represented by a smaller matrix.

Stolen from:
    https://github.com/lucidrains/nystrom-attention/blob/main/nystrom_attention/nystrom_attention.py

"""
from math import ceil
from typing import Optional, Tuple, Union

from einops import rearrange, reduce
import torch
from torch import einsum, nn, Tensor
from torch.nn import functional as F


def moore_penrose_inverse(x: Tensor, iters: int = 6) -> Tensor:
    """Moore-Penrose pseudoinverse."""
    x_abs = torch.abs(x)
    col = x_abs.sum(dim=-1)
    row = x_abs.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=x.device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        num_heads: int = 8,
        num_landmarks: int = 256,
        inverse_iter: int = 6,
        residual: bool = True,
        residual_conv_kernel: int = 13,
        eps: float = 1.0e-8,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.residual = None
        self.eps = eps
        self.num_heads = num_heads
        inner_dim = self.num_heads * dim_head
        self.num_landmarks = num_landmarks
        self.inverse_iter = inverse_iter
        self.scale = dim_head ** -0.5

        self.qkv_fn = nn.Linear(dim, 3 * inner_dim, bias=False)
        self.fc_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout_rate))

        if residual:
            self.residual = nn.Conv2d(
                in_channels=num_heads,
                out_channels=num_heads,
                kernel_size=(residual_conv_kernel, 1),
                padding=(residual_conv_kernel // 2, 0),
                groups=num_heads,
                bias=False,
            )

    @staticmethod
    def _pad_sequence(
        x: Tensor, mask: Optional[Tensor], n: int, m: int
    ) -> Tuple[Tensor, Tensor]:
        """Pad sequence."""
        padding = m - (n % m)
        x = F.pad(x, (0, 0, padding, 0), value=0)
        mask = F.pad(mask, (padding, 0), value=False) if mask is not None else mask
        return x, mask

    def _compute_landmarks(
        self, q: Tensor, k: Tensor, mask: Optional[Tensor], n: int, m: int
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Compute landmarks of the attention matrix."""
        divisor = ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=divisor)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=divisor)

        mask_landmarks = None
        if mask is not None:
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=divisor)
            divisor = mask_landmarks_sum[..., None] + self.eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks /= divisor
        k_landmarks /= divisor

        return q_landmarks, k_landmarks, mask_landmarks

    @staticmethod
    def _compute_similarities(
        q: Tensor,
        k: Tensor,
        q_landmarks: Tensor,
        k_landmarks: Tensor,
        mask: Optional[Tensor],
        mask_landmarks: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        if mask is not None and mask_landmarks is not None:
            mask_value = -torch.finfo(q.type).max
            sim1.masked_fill_(
                ~(mask[..., None] * mask_landmarks[..., None, :]), mask_value
            )
            sim2.masked_fill_(
                ~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value
            )
            sim3.masked_fill_(
                ~(mask_landmarks[..., None] * mask[..., None, :]), mask_value
            )

        return sim1, sim2, sim3

    def _nystrom_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        n: int,
        m: int,
        return_attn: bool,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q_landmarks, k_landmarks, mask_landmarks = self._compute_landmarks(
            q, k, mask, n, m
        )
        sim1, sim2, sim3 = self._compute_similarities(
            q, k, q_landmarks, k_landmarks, mask, mask_landmarks
        )

        # Compute attention
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_inverse(attn2, self.inverse_iter)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if return_attn:
            return out, attn1 @ attn2_inv @ attn3
        return out, None

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, return_attn: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute the Nystrom attention."""
        _, n, _, h, m = *x.shape, self.num_heads, self.num_landmarks
        if n % m != 0:
            x, mask = self._pad_sequence(x, mask, n, m)

        q, k, v = self.qkv_fn(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        out, attn = self._nystrom_attention(q, k, v, mask, n, m, return_attn)

        # Add depth-wise convolutional residual of values
        if self.residual is not None:
            out += self.residual(out)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.fc_out(out)
        out = out[:, -n:]

        if return_attn:
            return out, attn
        return out

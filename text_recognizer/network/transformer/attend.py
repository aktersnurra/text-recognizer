from typing import Optional
from collections import namedtuple

import torch
from torch import Tensor, einsum, nn
from einops import rearrange
import torch.nn.functional as F

Config = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


class Attend(nn.Module):
    def __init__(self, use_flash: bool) -> None:
        super().__init__()
        self.use_flash = use_flash
        self.cpu_cfg = Config(True, True, True)
        self.cuda_cfg = None
        if not torch.cuda.is_available():
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_cfg = Config(True, False, False)
        else:
            self.cuda_cfg = Config(False, True, True)

    def flash_attn(self, q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        cfg = self.cuda_cfg if q.is_cuda else self.cpu_cfg
        with torch.backends.cuda.sdp_kernel(**cfg._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return out

    def attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        b = q.shape[0]
        energy = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        mask_value = -torch.finfo(energy.dtype).max

        if mask is not None:
            energy = apply_input_mask(b, k, energy, mask, mask_value)

        if causal:
            energy = apply_causal_mask(energy, mask_value)

        attn = F.softmax(energy, dim=-1)
        attn = self.dropout(attn)
        return einsum("b h i j, b h j d -> b h i d", attn, v)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.use_flash:
            return self.flash_attn(q, k, v, causal)
        else:
            return self.attn(q, k, v, causal, mask)


def apply_input_mask(
    b: int,
    k: Tensor,
    energy: Tensor,
    mask: Optional[Tensor],
    mask_value: float,
) -> Tensor:
    """Applies an input mask."""
    k_mask = torch.ones((b, k.shape[-2]), device=energy.device).bool()
    q_mask = rearrange(mask, "b i -> b () i ()")
    k_mask = rearrange(k_mask, "b j -> b () () j")
    input_mask = q_mask * k_mask
    return energy.masked_fill_(~input_mask, mask_value)


def apply_causal_mask(
    energy: Tensor,
    mask_value: float,
) -> Tensor:
    """Applies a causal mask to the energy tensor."""
    i, j, device = *energy.shape[-2:], energy.device
    causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
    return energy.masked_fill(causal_mask, mask_value)

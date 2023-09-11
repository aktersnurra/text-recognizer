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

    def flash_attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        causal: bool,
    ) -> Tensor:
        cfg = self.cuda_cfg if q.is_cuda else self.cpu_cfg
        if causal:
            i, j, device = q.shape[-2], k.shape[-2], q.device
            causal_mask = create_causal_mask(i, j, device)
            mask = mask & ~causal_mask
            causal = False
        with torch.backends.cuda.sdp_kernel(**cfg._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=causal
            )
        return out

    def attn(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        causal: bool,
    ) -> Tensor:
        q.shape[0]
        weight = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        mask_value = -torch.finfo(weight.dtype).max

        if mask is not None:
            weight = weight.masked_fill(~mask, mask_value)

        if causal:
            i, j, device = weight.shape[-2:], weight.device
            causal_mask = create_causal_mask(i, j, device)
            weight = weight.masked_fill(causal_mask, mask_value)

        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        return einsum("b h i j, b h j d -> b h i d", weight, v)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")
        if self.use_flash:
            return self.flash_attn(q, k, v, mask, causal)
        else:
            return self.attn(q, k, v, mask, causal)


def create_causal_mask(
    i: int,
    j: int,
    device: torch.device,
) -> Tensor:
    """Applies a causal mask to the weight tensor."""
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

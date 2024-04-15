import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def l2_norm(t: Tensor, groups=1) -> Tensor:
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")

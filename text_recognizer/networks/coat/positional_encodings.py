"""Positional encodings for input sequence to transformer."""
from typing import Dict, Union, Tuple

from einops import rearrange
from loguru import logger
import torch
from torch import nn
from torch import Tensor


class RelativeEncoding(nn.Module):
    """Relative positional encoding."""
    def __init__(self, channels: int, heads: int, windows: Union[int, Dict[int, int]]) -> None:
        super().__init__()
        self.windows = {windows: heads} if isinstance(windows, int) else windows
        self.heads = list(self.windows.values())
        self.channel_heads = [head * channels for head in self.heads]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=head * channels,
                             out_channels=head * channels,
                             kernel_shape=window,
                             padding=window // 2,
                             dilation=1,
                             groups=head * channels,
            ) for window, head in self.windows.items()])

    def forward(self, q: Tensor, v: Tensor, shape: Tuple[int, int]) -> Tensor:
        """Applies relative positional encoding."""
        b, heads, hw, c = q.shape
        h, w = shape
        if hw != h * w:
            logger.exception(f"Query width {hw} neq to height x width {h * w}")
            raise ValueError
        
        v = rearrange(v, "b heads (h w) c -> b (heads c) h w", h=h, w=w)
        v = torch.split(v, self.channel_heads, dim=1)
        v = [conv(x) for conv, x in zip(self.convs, v)]
        v = torch.cat(v, dim=1)
        v = rearrange(v, "b (heads c) h w -> b heads (h w) c", heads=heads)

        encoding = q * v
        zeros = torch.zeros((b, heads, 1, c), dtype=q.dtype, layout=q.layout, device=q.device)
        encoding = torch.cat((zeros, encoding), dim=2)
        return encoding


class PositionalEncoding(nn.Module):
    """Convolutional positional encoding."""
    def __init__(self, dim: int, k: int = 3) -> None:
        super().__init__()
        self.encode = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=k, stride=1, padding=k//2, groups=dim)

    def forward(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        """Applies convolutional encoding."""
        _, hw, _ = x.shape
        h, w = shape

        if hw != h * w:
            logger.exception(f"Query width {hw} neq to height x width {h * w}")
            raise ValueError

        # Depthwise convolution.
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.encode(x) + x
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


        



            
            

        

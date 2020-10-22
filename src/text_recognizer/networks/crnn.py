"""LSTM with CTC for handwritten text recognition within a line."""
import importlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from loguru import logger
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import configure_backbone


class ConvolutionalRecurrentNetwork(nn.Module):
    """Network that takes a image of a text line and predicts tokens that are in the image."""

    def __init__(
        self,
        backbone: str,
        backbone_args: Dict = None,
        input_size: int = 128,
        hidden_size: int = 128,
        bidirectional: bool = False,
        num_layers: int = 1,
        num_classes: int = 80,
        patch_size: Tuple[int, int] = (28, 28),
        stride: Tuple[int, int] = (1, 14),
        recurrent_cell: str = "lstm",
    ) -> None:
        super().__init__()
        self.backbone_args = backbone_args or {}
        self.patch_size = patch_size
        self.stride = stride
        self.sliding_window = self._configure_sliding_window()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone = configure_backbone(backbone, backbone_args)
        self.bidirectional = bidirectional

        if recurrent_cell.upper() in ["LSTM", "GRU"]:
            recurrent_cell = getattr(nn, recurrent_cell)
        else:
            logger.warning(
                f"Option {recurrent_cell} not valid, defaulting to LSTM cell."
            )
            recurrent_cell = nn.LSTM

        self.rnn = recurrent_cell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

        decoder_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(in_features=decoder_size, out_features=num_classes),
            nn.LogSoftmax(dim=2),
        )

    def _configure_sliding_window(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Unfold(kernel_size=self.patch_size, stride=self.stride),
            Rearrange(
                "b (c h w) t -> b t c h w",
                h=self.patch_size[0],
                w=self.patch_size[1],
                c=1,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Converts images to sequence of patches, feeds them to a CNN, then predictions are made with an LSTM."""
        if len(x.shape) < 4:
            x = x[(None,) * (4 - len(x.shape))]
        x = self.sliding_window(x)

        # Rearrange from a sequence of patches for feedforward network.
        b, t = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w", b=b, t=t)
        x = self.backbone(x)

        # Avgerage pooling.
        x = reduce(x, "(b t) c h w -> t b c", "mean", b=b, t=t)

        # Sequence predictions.
        x, _ = self.rnn(x)

        # Sequence to classifcation layer.
        x = self.decoder(x)
        return x

"""LSTM with CTC for handwritten text recognition within a line."""
import importlib
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
import torch
from torch import nn
from torch import Tensor


class LineRecurrentNetwork(nn.Module):
    """Network that takes a image of a text line and predicts tokens that are in the image."""

    def __init__(
        self,
        encoder: str,
        encoder_args: Dict = None,
        flatten: bool = True,
        input_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 80,
        patch_size: Tuple[int, int] = (28, 28),
        stride: Tuple[int, int] = (1, 14),
    ) -> None:
        super().__init__()
        self.encoder_args = encoder_args or {}
        self.patch_size = patch_size
        self.stride = stride
        self.sliding_window = self._configure_sliding_window()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = self._configure_encoder(encoder)
        self.flatten = flatten
        self.fc = nn.Linear(in_features=self.input_size, out_features=self.hidden_size)
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=num_classes),
            nn.LogSoftmax(dim=2),
        )

    def _configure_encoder(self, encoder: str) -> Type[nn.Module]:
        network_module = importlib.import_module("text_recognizer.networks")
        encoder_ = getattr(network_module, encoder)
        return encoder_(**self.encoder_args)

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
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.sliding_window(x)

        # Rearrange from a sequence of patches for feedforward network.
        b, t = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w", b=b, t=t)
        x = self.encoder(x)

        # Avgerage pooling.
        x = reduce(x, "(b t) c h w -> t b c", "mean", b=b, t=t) if self.flatten else x

        # Linear layer between CNN and RNN
        x = self.fc(x)

        # Sequence predictions.
        x, _ = self.rnn(x)

        # Sequence to classifcation layer.
        x = self.decoder(x)
        return x

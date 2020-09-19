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


class LineRecurrentNetwork(nn.Module):
    """Network that takes a image of a text line and predicts tokens that are in the image."""

    def __init__(
        self,
        backbone: str,
        backbone_args: Dict = None,
        flatten: bool = True,
        input_size: int = 128,
        hidden_size: int = 128,
        bidirectional: bool = False,
        num_layers: int = 1,
        num_classes: int = 80,
        patch_size: Tuple[int, int] = (28, 28),
        stride: Tuple[int, int] = (1, 14),
    ) -> None:
        super().__init__()
        self.backbone_args = backbone_args or {}
        self.patch_size = patch_size
        self.stride = stride
        self.sliding_window = self._configure_sliding_window()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.backbone = self._configure_backbone(backbone)
        self.bidirectional = bidirectional
        self.flatten = flatten

        if self.flatten:
            self.fc = nn.Linear(
                in_features=self.input_size, out_features=self.hidden_size
            )

        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )

        decoder_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(in_features=decoder_size, out_features=num_classes),
            nn.LogSoftmax(dim=2),
        )

    def _configure_backbone(self, backbone: str) -> Type[nn.Module]:
        network_module = importlib.import_module("text_recognizer.networks")
        backbone_ = getattr(network_module, backbone)

        if "pretrained" in self.backbone_args:
            logger.info("Loading pretrained backbone.")
            checkpoint_file = Path(__file__).resolve().parents[
                2
            ] / self.backbone_args.pop("pretrained")

            # Loading state directory.
            state_dict = torch.load(checkpoint_file)
            network_args = state_dict["network_args"]
            weights = state_dict["model_state"]

            # Initializes the network with trained weights.
            backbone = backbone_(**network_args)
            backbone.load_state_dict(weights)
            if "freeze" in self.backbone_args and self.backbone_args["freeze"] is True:
                for params in backbone.parameters():
                    params.requires_grad = False

            return backbone
        else:
            return backbone_(**self.backbone_args)

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
        x = self.backbone(x)

        # Avgerage pooling.
        x = (
            self.fc(reduce(x, "(b t) c h w -> t b c", "mean", b=b, t=t))
            if self.flatten
            else rearrange(x, "(b t) h -> t b h", b=b, t=t)
        )

        # Sequence predictions.
        x, _ = self.rnn(x)

        # Sequence to classifcation layer.
        x = self.decoder(x)
        return x

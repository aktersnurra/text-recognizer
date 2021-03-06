"""Miscellaneous neural network functionality."""
import importlib
from pathlib import Path
from typing import Dict, Tuple, Type

from einops import rearrange
from loguru import logger
import torch
from torch import nn


def sliding_window(
    images: torch.Tensor, patch_size: Tuple[int, int], stride: Tuple[int, int]
) -> torch.Tensor:
    """Creates patches of an image.

    Args:
        images (torch.Tensor): A Torch tensor of a 4D image(s), i.e. (batch, channel, height, width).
        patch_size (Tuple[int, int]): The size of the patches to generate, e.g. 28x28 for EMNIST.
        stride (Tuple[int, int]): The stride of the sliding window.

    Returns:
        torch.Tensor: A tensor with the shape (batch, patches, height, width).

    """
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
    # Preform the sliding window, unsqueeze as the channel dimesion is lost.
    c = images.shape[1]
    patches = unfold(images)
    patches = rearrange(
        patches, "b (c h w) t -> b t c h w", c=c, h=patch_size[0], w=patch_size[1],
    )
    return patches


def activation_function(activation: str) -> Type[nn.Module]:
    """Returns the callable activation function."""
    activation_fns = nn.ModuleDict(
        [
            ["elu", nn.ELU(inplace=True)],
            ["gelu", nn.GELU()],
            ["glu", nn.GLU()],
            ["leaky_relu", nn.LeakyReLU(negative_slope=1.0e-2, inplace=True)],
            ["none", nn.Identity()],
            ["relu", nn.ReLU(inplace=True)],
            ["selu", nn.SELU(inplace=True)],
        ]
    )
    return activation_fns[activation.lower()]


def configure_backbone(backbone: str, backbone_args: Dict) -> Type[nn.Module]:
    """Loads a backbone network."""
    network_module = importlib.import_module("text_recognizer.networks")
    backbone_ = getattr(network_module, backbone)

    if "pretrained" in backbone_args:
        logger.info("Loading pretrained backbone.")
        checkpoint_file = Path(__file__).resolve().parents[2] / backbone_args.pop(
            "pretrained"
        )

        # Loading state directory.
        state_dict = torch.load(checkpoint_file)
        network_args = state_dict["network_args"]
        weights = state_dict["model_state"]

        freeze = False
        if "freeze" in backbone_args and backbone_args["freeze"] is True:
            backbone_args.pop("freeze")
            freeze = True
        network_args = backbone_args

        # Initializes the network with trained weights.
        backbone = backbone_(**network_args)
        backbone.load_state_dict(weights)
        if freeze:
            for params in backbone.parameters():
                params.requires_grad = False
    else:
        backbone_ = getattr(network_module, backbone)
        backbone = backbone_(**backbone_args)

    if "remove_layers" in backbone_args and backbone_args["remove_layers"] is not None:
        backbone = nn.Sequential(
            *list(backbone.children())[:][: -backbone_args["remove_layers"]]
        )

    return backbone

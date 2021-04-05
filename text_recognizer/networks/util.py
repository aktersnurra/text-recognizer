"""Miscellaneous neural network functionality."""
import importlib
from pathlib import Path
from typing import Dict, NamedTuple, Union, Type

from loguru import logger
import torch
from torch import nn


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


# def configure_backbone(backbone: Union[OmegaConf, NamedTuple]) -> Type[nn.Module]:
#     """Loads a backbone network."""
#     network_module = importlib.import_module("text_recognizer.networks")
#     backbone_class = getattr(network_module, backbone.type)
#
#     if "pretrained" in backbone.args:
#         logger.info("Loading pretrained backbone.")
#         checkpoint_file = Path(__file__).resolve().parents[2] / backbone.args.pop(
#             "pretrained"
#         )
#
#         # Loading state directory.
#         state_dict = torch.load(checkpoint_file)
#         network_args = state_dict["network_args"]
#         weights = state_dict["model_state"]
#
#         freeze = False
#         if "freeze" in backbone.args and backbone.args["freeze"] is True:
#             backbone.args.pop("freeze")
#             freeze = True
#
#         # Initializes the network with trained weights.
#         backbone_ = backbone_(**backbone.args)
#         backbone_.load_state_dict(weights)
#         if freeze:
#             for params in backbone_.parameters():
#                 params.requires_grad = False
#     else:
#         backbone_ = getattr(network_module, backbone.type)
#         backbone_ = backbone_(**backbone.args)
#
#     if "remove_layers" in backbone_args and backbone_args["remove_layers"] is not None:
#         backbone = nn.Sequential(
#             *list(backbone.children())[:][: -backbone_args["remove_layers"]]
#         )
#
#     return backbone

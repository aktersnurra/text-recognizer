"""Augmentations for training Barlow Twins."""
from omegaconf.dictconfig import DictConfig
from torch import Tensor

from text_recognizer.data.transforms.load_transform import load_transform


class BarlowTransform:
    """Applies two different transforms to input data."""

    def __init__(self, prim: DictConfig, bis: DictConfig) -> None:
        self.prim = load_transform(prim)
        self.bis = load_transform(bis)

    def __call__(self, data: Tensor) -> Tensor:
        """Applies two different augmentation on the input."""
        x_prim = self.prim(data)
        x_bis = self.bis(data)
        return x_prim, x_bis

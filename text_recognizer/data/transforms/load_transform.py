"""Load a config of transforms."""
from pathlib import Path
from typing import Callable

from loguru import logger as log
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate
import torchvision.transforms as T

TRANSFORM_DIRNAME = (
    Path(__file__).resolve().parents[3] / "training" / "conf" / "datamodule"
)


def _load_config(filepath: str) -> DictConfig:
    log.debug(f"Loading transforms from config: {filepath}")
    path = TRANSFORM_DIRNAME / Path(filepath)
    with open(path) as f:
        cfgs = OmegaConf.load(f)
    return cfgs


def _load_transform(transform: DictConfig) -> Callable:
    """Loads a transform."""
    if "ColorJitter" in transform._target_:
        return T.ColorJitter(brightness=list(transform.brightness))
    if transform.get("interpolation"):
        transform.interpolation = getattr(
            T.functional.InterpolationMode, transform.interpolation
        )
    return instantiate(transform, _recursive_=False)


def load_transform_from_file(filepath: str) -> T.Compose:
    """Loads transforms from a config."""
    cfgs = _load_config(filepath)
    transform = load_transform(cfgs)
    return transform


def load_transform(cfgs: DictConfig) -> T.Compose:
    transforms = []
    for cfg in cfgs.values():
        transform = _load_transform(cfg)
        transforms.append(transform)
    return T.Compose(transforms)

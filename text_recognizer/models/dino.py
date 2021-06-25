"""Dino: pretraining of models with self supervision."""
import copy
from functools import wraps, partial

import torch
from torch import nn
import torch.nn.funtional as F
import torchvision.transforms as T
import wandb

from text_recognizer.models.base import LitBaseModel


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn

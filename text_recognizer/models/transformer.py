"""PyTorch Lightning model for base Transformers."""
from typing import Dict, List, Optional, Union, Tuple, Type

import attr
import hydra
from omegaconf import DictConfig
from torch import nn, Tensor

from text_recognizer.data.emnist import emnist_mapping
from text_recognizer.data.mappings import AbstractMapping
from text_recognizer.models.metrics import CharacterErrorRate
from text_recognizer.models.base import BaseLitModel


@attr.s(auto_attribs=True)
class TransformerLitModel(BaseLitModel):
    """A PyTorch Lightning model for transformer networks."""

    mapping_config: DictConfig = attr.ib(converter=DictConfig)

    def __attrs_post_init__(self) -> None:
        self.mapping, ignore_tokens = self._configure_mapping()
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.network.predict(data)

    @staticmethod
    def _configure_mapping() -> Tuple[Type[AbstractMapping], List[int]]:
        """Configure mapping."""
        # TODO: Fix me!!!
        # Load config with hydra
        mapping, inverse_mapping, _ = emnist_mapping(["\n"])
        start_index = inverse_mapping["<s>"]
        end_index = inverse_mapping["<e>"]
        pad_index = inverse_mapping["<p>"]
        ignore_tokens = [start_index, end_index, pad_index]
        # TODO: add case for sentence pieces
        return mapping, ignore_tokens

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits = self.network(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, targets = batch

        logits = self.network(data, targets[:-1])
        loss = self.loss_fn(logits, targets[1:])
        self.log("val/loss", loss, prog_bar=True)

        pred = self.network.predict(data)
        self.val_cer(pred, targets)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch
        pred = self.network.predict(data)
        self.test_cer(pred, targets)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

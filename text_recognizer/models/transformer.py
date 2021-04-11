"""PyTorch Lightning model for base Transformers."""
from typing import Dict, List, Optional, Union, Tuple, Type

from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import wandb

from text_recognizer.data.emnist import emnist_mapping
from text_recognizer.models.metrics import CharacterErrorRate
from text_recognizer.models.base import LitBaseModel


class LitTransformerModel(LitBaseModel):
    """A PyTorch Lightning model for transformer networks."""

    def __init__(
        self,
        network: Type[nn.Module],
        optimizer: Union[DictConfig, Dict],
        lr_scheduler: Union[DictConfig, Dict],
        criterion: Union[DictConfig, Dict],
        monitor: str = "val_loss",
        mapping: Optional[List[str]] = None,
    ) -> None:
        super().__init__(network, optimizer, lr_scheduler, criterion, monitor)
        self.mapping, ignore_tokens = self.configure_mapping(mapping)
        self.val_cer = CharacterErrorRate(ignore_tokens)
        self.test_cer = CharacterErrorRate(ignore_tokens)

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.network.predict(data)

    @staticmethod
    def configure_mapping(mapping: Optional[List[str]]) -> Tuple[List[str], List[int]]:
        """Configure mapping."""
        # TODO: Fix me!!!
        mapping, inverse_mapping, _ = emnist_mapping()
        start_index = inverse_mapping["<s>"]
        end_index = inverse_mapping["<e>"]
        pad_index = inverse_mapping["<p>"]
        ignore_tokens = [start_index, end_index, pad_index]
        # TODO: add case for sentence pieces
        return mapping, ignore_tokens

    def _log_prediction(self, data: Tensor, pred: Tensor) -> None:
        """Logs prediction on image with wandb."""
        pred_str = "".join(
            self.mapping[i] for i in pred[0].tolist() if i != 3
        )  # pad index is 3
        try:
            self.logger.experiment.log(
                {"val_pred_examples": [wandb.Image(data[0], caption=pred_str)]}
            )
        except AttributeError:
            pass

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits = self.network(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, targets = batch

        logits = self.network(data, targets[:-1])
        loss = self.loss_fn(logits, targets[1:])
        self.log("val_loss", loss, prog_bar=True)

        pred = self.network.predict(data)
        self._log_prediction(data, pred)
        self.val_cer(pred, targets)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch
        pred = self.network.predict(data)
        self._log_prediction(data, pred)
        self.test_cer(pred, targets)
        self.log("test_cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)

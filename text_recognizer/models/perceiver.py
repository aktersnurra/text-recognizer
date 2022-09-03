"""Lightning model for base Perceiver."""
from typing import Optional, Tuple, Type

from omegaconf import DictConfig
import torch
from torch import nn, Tensor

from text_recognizer.data.mappings import EmnistMapping
from text_recognizer.models.base import LitBase
from text_recognizer.models.metrics import CharacterErrorRate


class LitPerceiver(LitBase):
    """A PyTorch Lightning model for transformer networks."""

    def __init__(
        self,
        network: Type[nn.Module],
        loss_fn: Type[nn.Module],
        optimizer_config: DictConfig,
        lr_scheduler_config: Optional[DictConfig],
        mapping: EmnistMapping,
        max_output_len: int = 682,
        start_token: str = "<s>",
        end_token: str = "<e>",
        pad_token: str = "<p>",
    ) -> None:
        super().__init__(
            network, loss_fn, optimizer_config, lr_scheduler_config, mapping
        )
        self.max_output_len = max_output_len
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.start_index = int(self.mapping.get_index(self.start_token))
        self.end_index = int(self.mapping.get_index(self.end_token))
        self.pad_index = int(self.mapping.get_index(self.pad_token))
        self.ignore_indices = set([self.start_index, self.end_index, self.pad_index])
        self.val_cer = CharacterErrorRate(self.ignore_indices)
        self.test_cer = CharacterErrorRate(self.ignore_indices)

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.predict(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits = self.network(data)
        loss = self.loss_fn(logits, targets)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, targets = batch
        preds = self.predict(data)
        self.val_acc(preds, targets)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(preds, targets)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch

        # Compute the text prediction.
        pred = self(data)
        self.test_cer(pred, targets)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc(pred, targets)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        return self.network(x).argmax(dim=1)

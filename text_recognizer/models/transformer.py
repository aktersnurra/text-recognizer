"""Lightning model for base Transformers."""
from typing import Callable, Optional, Sequence, Tuple, Type
from text_recognizer.models.greedy_decoder import GreedyDecoder

import torch
from omegaconf import DictConfig
from torch import nn, Tensor
from torchmetrics import CharErrorRate, WordErrorRate

from text_recognizer.data.tokenizer import Tokenizer
from text_recognizer.models.base import LitBase


class LitTransformer(LitBase):
    """A PyTorch Lightning model for transformer networks."""

    def __init__(
        self,
        network: Type[nn.Module],
        loss_fn: Type[nn.Module],
        optimizer_config: DictConfig,
        tokenizer: Tokenizer,
        decoder: Callable = GreedyDecoder,
        lr_scheduler_config: Optional[DictConfig] = None,
        max_output_len: int = 682,
    ) -> None:
        super().__init__(
            network,
            loss_fn,
            optimizer_config,
            lr_scheduler_config,
            tokenizer,
        )
        self.max_output_len = max_output_len
        self.val_cer = CharErrorRate()
        self.test_cer = CharErrorRate()
        self.val_wer = WordErrorRate()
        self.test_wer = WordErrorRate()
        self.decoder = decoder

    def forward(self, data: Tensor) -> Tensor:
        """Autoregressive forward pass."""
        return self.predict(data)

    def teacher_forward(self, data: Tensor, targets: Tensor) -> Tensor:
        """Non-autoregressive forward pass."""
        return self.network(data, targets)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits = self.teacher_forward(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, targets = batch

        logits = self.teacher_forward(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        preds = self.predict(data)
        pred_text, target_text = self._to_tokens(preds), self._to_tokens(targets)

        self.val_acc(preds, targets)
        self.val_cer(pred_text, target_text)
        self.val_wer(pred_text, target_text)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/wer", self.val_wer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch

        logits = self.teacher_forward(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        preds = self(data)
        pred_text, target_text = self._to_tokens(preds), self._to_tokens(targets)

        self.test_acc(preds, targets)
        self.test_cer(pred_text, target_text)
        self.test_wer(pred_text, target_text)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/wer", self.test_wer, on_step=False, on_epoch=True, prog_bar=True)

    def _to_tokens(
        self,
        indecies: Tensor,
    ) -> Sequence[str]:
        return [self.tokenizer.decode(i) for i in indecies]

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        return self.decoder(x)

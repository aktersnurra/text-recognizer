"""Lightning model for transformer networks."""
from typing import Callable, Optional, Tuple, Type

import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torchmetrics import CharErrorRate, WordErrorRate

from text_recognizer.data.tokenizer import Tokenizer
from text_recognizer.decoder.greedy_decoder import GreedyDecoder

from .base import LitBase


class LitTransformer(LitBase):
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
        logits = self.network(data, targets)  # [B, N, C]
        return logits.permute(0, 2, 1)  # [B, C, N]

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> dict:
        """Training step."""
        data, targets = batch
        logits = self.teacher_forward(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])

        self.log("train/loss", loss, prog_bar=True)

        outputs = {"loss": loss}

        if self.is_logged_batch():
            preds, gts = self.tokenizer.decode_logits(
                logits
            ), self.tokenizer.batch_decode(targets)
            outputs.update({"predictions": preds, "ground_truths": gts})

        return outputs

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> dict:
        """Validation step."""
        data, targets = batch
        preds = self(data)
        preds, gts = self.tokenizer.batch_decode(preds), self.tokenizer.batch_decode(
            targets
        )

        self.val_cer(preds, gts)
        self.val_wer(preds, gts)

        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/wer", self.val_wer, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {}
        self.add_on_first_batch(
            {"predictions": preds, "ground_truths": gts}, outputs, batch_idx
        )
        return outputs

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> dict:
        """Test step."""
        data, targets = batch
        preds = self(data)
        preds, gts = self.tokenizer.batch_decode(preds), self.tokenizer.batch_decode(
            targets
        )

        self.test_cer(preds, gts)
        self.test_wer(preds, gts)

        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/wer", self.test_wer, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {}
        self.add_on_first_batch(
            {"predictions": preds, "ground_truths": gts}, outputs, batch_idx
        )
        return outputs

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        return self.decoder(x)

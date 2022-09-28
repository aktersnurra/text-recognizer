"""Lightning model for base Transformers."""
from collections.abc import Sequence
from typing import Optional, Tuple, Type

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

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.predict(data)

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
        preds = self.predict(data)
        pred_text, target_text = self._get_text(preds), self._get_text(targets)
        self.val_acc(preds, targets)
        self.val_cer(pred_text, target_text)
        self.val_wer(pred_text, target_text)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/wer", self.val_wer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch

        # Compute the text prediction.
        preds = self(data)
        pred_text, target_text = self._get_text(preds), self._get_text(targets)
        self.test_acc(preds, targets)
        self.test_cer(pred_text, target_text)
        self.test_wer(pred_text, target_text)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/wer", self.test_wer, on_step=False, on_epoch=True, prog_bar=True)

    def _get_text(
        self,
        xs: Tensor,
    ) -> Tuple[Sequence[str], Sequence[str]]:
        return [self.tokenizer.decode(x) for x in xs]

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Predicts text in image.

        Args:
            x (Tensor): Image(s) to extract text from.

        Shapes:
            - x: :math: `(B, H, W)`
            - output: :math: `(B, S)`

        Returns:
            Tensor: A tensor of token indices of the predictions from the model.
        """
        start_index = self.tokenizer.start_index
        end_index = self.tokenizer.end_index
        pad_index = self.tokenizer.pad_index
        bsz = x.shape[0]

        # Encode image(s) to latent vectors.
        z = self.network.encode(x)

        # Create a placeholder matrix for storing outputs from the network
        output = torch.ones((bsz, self.max_output_len), dtype=torch.long).to(x.device)
        output[:, 0] = start_index

        for Sy in range(1, self.max_output_len):
            context = output[:, :Sy]  # (B, Sy)
            logits = self.network.decode(z, context)  # (B, C, Sy)
            tokens = torch.argmax(logits, dim=1)  # (B, Sy)
            output[:, Sy : Sy + 1] = tokens[:, -1:]

            # Early stopping of prediction loop if token is end or padding token.
            if (
                (output[:, Sy - 1] == end_index) | (output[:, Sy - 1] == pad_index)
            ).all():
                break

        # Set all tokens after end token to pad token.
        for Sy in range(1, self.max_output_len):
            idx = (output[:, Sy - 1] == end_index) | (output[:, Sy - 1] == pad_index)
            output[idx, Sy] = pad_index

        return output

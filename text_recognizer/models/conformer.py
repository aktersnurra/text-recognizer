"""Lightning Conformer model."""
import itertools
from typing import Optional, Tuple, Type

from omegaconf import DictConfig
import torch
from torch import nn, Tensor

from text_recognizer.data.mappings import AbstractMapping
from text_recognizer.models.base import LitBase
from text_recognizer.models.metrics import CharacterErrorRate
from text_recognizer.models.util import first_element


class LitConformer(LitBase):
    """A PyTorch Lightning model for transformer networks."""

    def __init__(
        self,
        network: Type[nn.Module],
        loss_fn: Type[nn.Module],
        optimizer_configs: DictConfig,
        lr_scheduler_configs: Optional[DictConfig],
        mapping: Type[AbstractMapping],
        max_output_len: int = 451,
        start_token: str = "<s>",
        end_token: str = "<e>",
        pad_token: str = "<p>",
        blank_token: str = "<b>",
    ) -> None:
        super().__init__(
            network, loss_fn, optimizer_configs, lr_scheduler_configs, mapping
        )
        self.max_output_len = max_output_len
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.blank_token = blank_token
        self.start_index = int(self.mapping.get_index(self.start_token))
        self.end_index = int(self.mapping.get_index(self.end_token))
        self.pad_index = int(self.mapping.get_index(self.pad_token))
        self.blank_index = int(self.mapping.get_index(self.blank_token))
        self.ignore_indices = set(
            [self.start_index, self.end_index, self.pad_index, self.blank_index]
        )
        self.val_cer = CharacterErrorRate(self.ignore_indices)
        self.test_cer = CharacterErrorRate(self.ignore_indices)

    @torch.no_grad()
    def predict(self, x: Tensor) -> str:
        """Predicts a sequence of characters."""
        logits = self(x)
        logprobs = torch.log_softmax(logits, dim=1)
        return self.decode(logprobs, self.max_output_len)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits = self(data)
        logprobs = torch.log_softmax(logits, dim=1)
        B, S, _ = logprobs.shape
        input_length = torch.ones(B).type_as(logprobs).int() * S
        target_length = first_element(targets, self.pad_index).type_as(targets)
        loss = self.loss_fn(
            logprobs.permute(1, 0, 2), targets, input_length, target_length
        )
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, targets = batch
        logits = self(data)
        logprobs = torch.log_softmax(logits, dim=1)
        B, S, _ = logprobs.shape
        input_length = torch.ones(B).type_as(logprobs).int() * S
        target_length = first_element(targets, self.pad_index).type_as(targets)
        loss = self.loss_fn(
            logprobs.permute(1, 0, 2), targets, input_length, target_length
        )
        self.log("val/loss", loss)
        preds = self.decode(logprobs, targets.shape[1])
        self.val_acc(preds, targets)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(preds, targets)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch
        logits = self(data)
        logprobs = torch.log_softmax(logits, dim=1)
        preds = self.decode(logprobs, targets.shape[1])
        self.val_acc(preds, targets)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.val_cer(preds, targets)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def decode(self, logprobs: Tensor, max_length: int) -> Tensor:
        """Greedly decodes a log prob sequence.

        Args:
            logprobs (Tensor): Log probabilities.
            max_length (int): Max length of a sequence.

        Shapes:
            - x: :math: `(B, T, C)`
            - output: :math: `(B, T)`

        Returns:
            Tensor: A predicted sequence of characters.
        """
        B = logprobs.shape[0]
        argmax = logprobs.argmax(2)
        decoded = torch.ones((B, max_length)).type_as(logprobs).int() * self.pad_index
        for i in range(B):
            seq = [
                b
                for b, _ in itertools.groupby(argmax[i].tolist())
                if b != self.blank_index
            ][:max_length]
            for j, c in enumerate(seq):
                decoded[i, j] = c
        return decoded

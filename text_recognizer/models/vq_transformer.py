"""PyTorch Lightning model for base Transformers."""
from typing import Tuple, Type

import attr
import torch
from torch import Tensor

from text_recognizer.data.base_mapping import AbstractMapping
from text_recognizer.models.transformer import TransformerLitModel


@attr.s(auto_attribs=True, eq=False)
class VqTransformerLitModel(TransformerLitModel):
    """A PyTorch Lightning model for transformer networks."""

    mapping: Type[AbstractMapping] = attr.ib()
    alpha: float = attr.ib(default=1.0)

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.predict(data)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits, commitment_loss = self.network(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:]) + self.alpha * commitment_loss
        self.log("train/loss", loss)
        self.log("train/commitment_loss", commitment_loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, targets = batch
        logits, commitment_loss = self.network(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:]) + self.alpha * commitment_loss
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/commitment_loss", commitment_loss)

        # Get the token prediction.
        # pred = self(data)
        # self.val_cer(pred, targets)
        # self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)
        # self.test_acc(pred, targets)
        # self.log("val/acc", self.test_acc, on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, targets = batch
        pred = self(data)
        self.test_cer(pred, targets)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc(pred, targets)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

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
        bsz = x.shape[0]

        # Encode image(s) to latent vectors.
        z, _ = self.network.encode(x)

        # Create a placeholder matrix for storing outputs from the network
        output = torch.ones((bsz, self.max_output_len), dtype=torch.long).to(x.device)
        output[:, 0] = self.start_index

        for Sy in range(1, self.max_output_len):
            context = output[:, :Sy]  # (B, Sy)
            logits = self.network.decode(z, context)  # (B, C, Sy)
            tokens = torch.argmax(logits, dim=1)  # (B, Sy)
            output[:, Sy : Sy + 1] = tokens[:, -1:]

            # Early stopping of prediction loop if token is end or padding token.
            if (
                (output[:, Sy - 1] == self.end_index)
                | (output[:, Sy - 1] == self.pad_index)
            ).all():
                break

        # Set all tokens after end token to pad token.
        for Sy in range(1, self.max_output_len):
            idx = (output[:, Sy - 1] == self.end_index) | (
                output[:, Sy - 1] == self.pad_index
            )
            output[idx, Sy] = self.pad_index

        return output

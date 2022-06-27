"""Lightning model for Vector Quantized Transformers."""
from typing import Optional, Tuple, Type

from omegaconf import DictConfig
import torch
from torch import nn, Tensor

from text_recognizer.data.mappings import EmnistMapping
from text_recognizer.models.transformer import LitTransformer


class LitVqTransformer(LitTransformer):
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
        vq_loss_weight: float = 0.1,
    ) -> None:
        super().__init__(
            network,
            loss_fn,
            optimizer_config,
            lr_scheduler_config,
            mapping,
            max_output_len,
            start_token,
            end_token,
            pad_token,
        )
        self.vq_loss_weight = vq_loss_weight

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, targets = batch
        logits, vq_loss = self.network(data, targets[:, :-1])
        loss = self.loss_fn(logits, targets[:, 1:])
        total_loss = loss + self.vq_loss_weight * vq_loss
        self.log("train/vq_loss", vq_loss)
        self.log("train/loss", loss)
        self.log("train/total_loss", total_loss)
        return total_loss

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
        pred = self(data)
        self.test_cer(pred, targets)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc(pred, targets)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

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

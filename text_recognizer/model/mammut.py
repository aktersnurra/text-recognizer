"""Lightning model for transformer networks."""
from typing import Callable, Optional, Tuple, Type
from text_recognizer.network.mammut import MaMMUT

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import einsum, nn, Tensor
from torchmetrics import CharErrorRate, WordErrorRate
import torch.nn.functional as F

from text_recognizer.decoder.greedy_decoder import GreedyDecoder
from text_recognizer.data.tokenizer import Tokenizer
from .base import LitBase


class LitMaMMUT(LitBase):
    def __init__(
        self,
        network: MaMMUT,
        loss_fn: Type[nn.Module],
        optimizer_config: DictConfig,
        tokenizer: Tokenizer,
        decoder: Callable = GreedyDecoder,
        lr_scheduler_config: Optional[DictConfig] = None,
        max_output_len: int = 682,
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
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
        self.contrastive_loss = F.cross_entropy
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.temperature = nn.Parameter(Tensor([1.0]))

    def forward(self, data: Tensor) -> Tensor:
        """Autoregressive forward pass."""
        return self.predict(data)

    def to_caption_loss(self, logits: Tensor, text: Tensor) -> Tensor:
        caption_loss = self.loss_fn(logits, text[:, 1:])
        return self.caption_loss_weight * caption_loss

    def to_contrastive_loss(
        self, image_embeddings: Tensor, text_embeddings: Tensor
    ) -> Tensor:
        b, device = image_embeddings.shape[0], image_embeddings.device
        image_latents, text_latents = self.network.to_latents(
            image_embeddings, text_embeddings
        )
        sim = einsum("i d, j d -> i j", text_latents, image_latents)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(b, device=device)
        contrastive_loss = (
            F.cross_entropy(sim, contrastive_labels)
            + F.cross_entropy(sim.t(), contrastive_labels)
        ) / 2
        return self.contrastive_loss_weight * contrastive_loss

    def teacher_forward(self, images: Tensor, text: Tensor) -> Tuple[Tensor, Tensor]:
        """Non-autoregressive forward pass."""
        text_embeddings = self.network.to_text_cls_features(text[:, :-1])
        image_embeddings, image_features = self.network.to_image_features(images)
        logits = self.network.decode(text[:, :-1], image_features)
        logits = rearrange(logits, "b n c -> b c n")

        caption_loss = self.to_caption_loss(logits, text)
        contrastive_loss = self.to_contrastive_loss(image_embeddings, text_embeddings)

        self.log("train/caption_loss", caption_loss)
        self.log("train/contrastive_loss", contrastive_loss)

        return logits, caption_loss + contrastive_loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> dict:
        """Training step."""
        data, targets = batch
        logits, loss = self.teacher_forward(data, targets)
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

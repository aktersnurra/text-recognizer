"""PyTorch Lightning model for base Transformers."""
from typing import Any, Dict, Union, Tuple, Type

from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import wandb

from text_recognizer.models.base import LitBaseModel


class LitVQVAEModel(LitBaseModel):
    """A PyTorch Lightning model for transformer networks."""

    def __init__(
        self,
        network: Type[nn.Module],
        optimizer: Union[DictConfig, Dict],
        lr_scheduler: Union[DictConfig, Dict],
        criterion: Union[DictConfig, Dict],
        monitor: str = "val_loss",
        *args: Any,
        **kwargs: Dict,
    ) -> None:
        super().__init__(network, optimizer, lr_scheduler, criterion, monitor)

    def forward(self, data: Tensor) -> Tensor:
        """Forward pass with the transformer network."""
        return self.network.predict(data)

    def _log_prediction(
        self, data: Tensor, reconstructions: Tensor, title: str
    ) -> None:
        """Logs prediction on image with wandb."""
        try:
            self.logger.experiment.log(
                {title: [wandb.Image(data[0]), wandb.Image(reconstructions[0]),]}
            )
        except AttributeError:
            pass

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        data, _ = batch
        reconstructions, vq_loss = self.network(data)
        loss = self.loss_fn(reconstructions, data)
        loss += vq_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        data, _ = batch
        reconstructions, vq_loss = self.network(data)
        loss = self.loss_fn(reconstructions, data)
        loss += vq_loss
        self.log("val_loss", loss, prog_bar=True)
        title = "val_pred_examples"
        self._log_prediction(data, reconstructions, title)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        data, _ = batch
        reconstructions, vq_loss = self.network(data)
        loss = self.loss_fn(reconstructions, data)
        loss += vq_loss
        title = "test_pred_examples"
        self._log_prediction(data, reconstructions, title)

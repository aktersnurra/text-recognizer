"""Script to run experiments."""
from typing import List, Optional, Type

import hydra
import loguru.logger as log
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from torch import nn

import utils


def run(config: DictConfig) -> Optional[float]:
    """Runs experiment."""
    utils.configure_logging(config.logging)
    log.info("Starting experiment...")

    if config.get("seed"):
        seed_everything(config.seed)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info(f"Instantiating network <{config.network._target_}>")
    network: nn.Module = hydra.utils.instantiate(config.network, **datamodule.config())

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        network=network,
        criterion=config.criterion,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        _recursive_=False,
    )

    # Load callback and logger.
    callbacks: List[Type[Callback]] = utils.configure_callbacks(config)
    logger: List[Type[LightningLoggerBase]] = utils.configure_logger(config)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Log hyperparameters
    log.info("Logging hyperparameters")
    utils.log_hyperparameters(config=config, model=model, trainer=trainer)

    if config.debug:
        log.info("Fast development run...")
        trainer.fit(model, datamodule=datamodule)
        return None

    if config.tune:
        log.info("Tuning hyperparameters...")
        trainer.tune(model, datamodule=datamodule)

    if config.train:
        log.info("Training network...")
        trainer.fit(model, datamodule=datamodule)

    if config.test:
        log.info("Testing network...")
        trainer.test(model, datamodule=datamodule)

    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    utils.finish(trainer)

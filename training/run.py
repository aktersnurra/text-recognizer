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

from utils import configure_logging


def configure_callbacks(
    config: DictConfig,
) -> List[Type[Callback]]:
    """Configures lightning callbacks."""
    callbacks = []
    if config.get("callbacks"):
        for callback_config in config.callbacks.values():
            if config.get("_target_"):
                log.info(f"Instantiating callback <{callback_config._target_}>")
                callbacks.append(hydra.utils.instantiate(callback_config))
    return callbacks


def configure_logger(config: DictConfig) -> List[Type[LightningLoggerBase]]:
    logger = []
    if config.get("logger"):
        for logger_config in config.logger.values():
            if config.get("_target_"):
                log.info(f"Instantiating callback <{logger_config._target_}>")
                logger.append(hydra.utils.instantiate(logger_config))
    return logger


def run(config: DictConfig) -> Optional[float]:
    """Runs experiment."""
    configure_logging(config.logging)
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
    callbacks = configure_callbacks(config)
    logger = configure_logger(config)

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Log hyperparameters

    if config.debug:
        log.info("Fast development run...")
        trainer.fit(model, datamodule=datamodule)
        return None

    if config.tune:
        log.info("Tuning learning rate and batch size...")
        trainer.tune(model, datamodule=datamodule)

    if config.train:
        log.info("Training network...")
        trainer.fit(model, datamodule=datamodule)

    if config.test:
        log.info("Testing network...")
        trainer.test(model, datamodule=datamodule)

    # Make sure everything closes properly

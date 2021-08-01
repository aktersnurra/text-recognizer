"""Util functions for training with hydra and pytorch lightning."""
from typing import Any, List, Type
import warnings

import hydra
import loguru.logger as log
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
import wandb


@rank_zero_only
def configure_logging(config: DictConfig) -> None:
    """Configure the loguru logger for output to terminal and disk."""
    # Remove default logger to get tqdm to work properly.
    log.remove()
    log.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=config.logging)


def configure_callbacks(config: DictConfig,) -> List[Type[Callback]]:
    """Configures Lightning callbacks."""
    callbacks = []
    if config.get("callbacks"):
        for callback_config in config.callbacks.values():
            if config.get("_target_"):
                log.info(f"Instantiating callback <{callback_config._target_}>")
                callbacks.append(hydra.utils.instantiate(callback_config))
    return callbacks


def configure_logger(config: DictConfig) -> List[Type[LightningLoggerBase]]:
    """Configures Lightning loggers."""
    logger = []
    if config.get("logger"):
        for logger_config in config.logger.values():
            if config.get("_target_"):
                log.info(f"Instantiating callback <{logger_config._target_}>")
                logger.append(hydra.utils.instantiate(logger_config))
    return logger


def extras(config: DictConfig) -> None:
    """Sets optional utilities."""
    # Enable adding new keys.
    OmegaConf.set_struct(config, False)

    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        # Debuggers do not like GPUs and multiprocessing.
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # Force multi-gpu friendly config.
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(
            f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>"
        )
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # Disable adding new keys to config
    OmegaConf.set_struct(config, True)


def empty(*args: Any, **kwargs: Any) -> None:
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig, model: LightningModule, trainer: Trainer,
) -> None:
    """This method saves hyperparameters with the logger."""
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(logger: List[Type[LightningLoggerBase]],) -> None:
    """Makes sure everything closed properly."""
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()

"""Util functions for training with hydra and pytorch lightning."""
import warnings
from typing import List, Type

import hydra
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

import wandb


def save_config(config: DictConfig) -> None:
    """Save config to experiment directory."""
    with open("config.yaml", "w") as f:
        OmegaConf.save(config, f=f)


def print_config(config: DictConfig) -> None:
    """Prints config."""
    print(OmegaConf.to_yaml(config))


@rank_zero_only
def configure_logging(config: DictConfig) -> None:
    """Configure the loguru logger for output to terminal and disk."""
    # Remove default logger to get tqdm to work properly.
    log.remove()
    log.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=config.logging)


def configure_callbacks(
    config: DictConfig,
) -> List[Type[Callback]]:
    """Configures Lightning callbacks."""

    def load_callback(callback_config: DictConfig) -> Type[Callback]:
        log.info(f"Instantiating callback <{callback_config._target_}>")
        return hydra.utils.instantiate(callback_config)

    def load_callbacks(callback_configs: DictConfig) -> List[Type[Callback]]:
        callbacks = []
        for callback_config in callback_configs.values():
            if callback_config.get("_target_"):
                callbacks.append(load_callback(callback_config))
            else:
                callbacks += load_callbacks(callback_config)
        return callbacks

    if config.get("callbacks"):
        callbacks = load_callbacks(config.callbacks)
    return callbacks


def configure_logger(config: DictConfig) -> List[Type[Logger]]:
    """Configures Lightning loggers."""

    def load_logger(logger_config: DictConfig) -> Type[Logger]:
        log.info(f"Instantiating logger <{logger_config._target_}>")
        return hydra.utils.instantiate(logger_config)

    logger = []
    if config.get("logger"):
        for logger_config in config.logger.values():
            if logger_config.get("_target_"):
                logger.append(load_logger(logger_config))
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
        if config.trainer.get("precision"):
            config.trainer.precision = 32
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # Disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: LightningModule,
    trainer: Trainer,
) -> None:
    """This method saves hyperparameters with the logger."""
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    hparams["callbacks"] = config.get("callbacks")
    hparams["tags"] = config.get("tags")
    hparams["ckpt_path"] = config.get("ckpt_path")
    hparams["seed"] = config.get("seed")

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    logger: List[Type[Logger]],
) -> None:
    """Makes sure everything closed properly."""
    for lg in logger:
        if isinstance(lg, WandbLogger):
            wandb.finish()

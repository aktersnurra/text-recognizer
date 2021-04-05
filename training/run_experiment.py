"""Script to run experiments."""
from datetime import datetime
import importlib
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union, Type

import click
from loguru import logger
import numpy as np
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch
from torch import nn
from torchsummary import summary
from tqdm import tqdm
import wandb


SEED = 4711
EXPERIMENTS_DIRNAME = Path(__file__).parents[0].resolve() / "experiments"


def _configure_logging(log_dir: Optional[Path], verbose: int = 0) -> None:
    """Configure the loguru logger for output to terminal and disk."""

    def _get_level(verbose: int) -> int:
        """Sets the logger level."""
        levels = {0: 40, 1: 20, 2: 10}
        verbose = verbose if verbose <= 2 else 2
        return levels[verbose]

    # Have to remove default logger to get tqdm to work properly.
    logger.remove()

    # Fetch verbosity level.
    level = _get_level(verbose)

    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=level)
    if log_dir is not None:
        logger.add(
            str(log_dir / "train.log"),
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def _import_class(module_and_class_name: str) -> type:
    """Import class from module."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _configure_pl_callbacks(args: List[Union[OmegaConf, NamedTuple]]) -> List[Type[pl.callbacks.Callback]]:
    """Configures PyTorch Lightning callbacks."""
    pl_callbacks = [
        getattr(pl.callbacks, callback.type)(**callback.args) for callback in args
    ]
    return pl_callbacks


def _configure_wandb_callback(
    network: Type[nn.Module], args: Dict
) -> pl.loggers.WandbLogger:
    """Configures wandb logger."""
    pl_logger = pl.loggers.WandbLogger()
    pl_logger.watch(network)
    pl_logger.log_hyperparams(vars(args))
    return pl_logger


def _save_best_weights(
    callbacks: List[Type[pl.callbacks.Callback]], use_wandb: bool
) -> None:
    """Saves the best model."""
    model_checkpoint_callback = next(
        callback
        for callback in callbacks
        if isinstance(callback, pl.callbacks.ModelCheckpoint)
    )
    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Best model saved at: {best_model_path}")
        if use_wandb:
            logger.info("Uploading model to W&B...")
            wandb.save(best_model_path)


def run(path: str, train: bool, test: bool, tune: bool, use_wandb: bool) -> None:
    """Runs experiment."""
    logger.info("Starting experiment...")

    # Seed everything in the experiment
    logger.info(f"Seeding everthing with seed={SEED}")
    pl.utilities.seed.seed_everything(SEED)

    # Load config.
    logger.info(f"Loading config from: {path}")
    config = OmegaConf.load(path)

    # Load classes
    data_module_class = _import_class(f"text_recognizer.data.{config.data.type}")
    network_class = _import_class(f"text_recognizer.networks.{config.network.type}")
    lit_model_class = _import_class(f"text_recognizer.models.{config.model.type}")

    # Initialize data object and network.
    data_module = data_module_class(**config.data.args)
    network = network_class(**config.network.args)

    # Load callback and logger
    callbacks = _configure_pl_callbacks(config.callbacks)
    pl_logger = (
        _configure_wandb_callback(network, config.network.args)
        if use_wandb
        else pl.logger.TensorBoardLogger("training/logs")
    )

    # Checkpoint
    if config.load_checkpoint is not None:
        logger.info(
            f"Loading network weights from checkpoint: {config.load_checkpoint}"
        )
        lit_model = lit_model_class.load_from_checkpoint(
            config.load_checkpoint, network=network, **config.model.args
        )
    else:
        lit_model = lit_model_class(**config.model.args)

    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=pl_logger,
        weigths_save_path="training/logs",
    )

    if tune:
        logger.info(f"Tuning learning rate and batch size...")
        trainer.tune(lit_model, datamodule=data_module)

    if train:
        logger.info(f"Training network...")
        trainer.fit(lit_model, datamodule=data_module)

    if test:
        logger.info(f"Testing network...")
        trainer.test(lit_model, datamodule=data_module)

    _save_best_weights(callbacks, use_wandb)


@click.command()
@click.option("-f", "--experiment_config", type=str, help="Path to experiment config.")
@click.option("--use_wandb", is_flag=True, help="If true, do use wandb for logging.")
@click.option(
    "--tune", is_flag=True, help="If true, tune hyperparameters for training."
)
@click.option("--train", is_flag=True, help="If true, train the model.")
@click.option("--test", is_flag=True, help="If true, test the model.")
@click.option("-v", "--verbose", count=True)
def cli(
    experiment_config: str,
    use_wandb: bool,
    tune: bool,
    train: bool,
    test: bool,
    verbose: int,
) -> None:
    """Run experiment."""
    _configure_logging(None, verbose=verbose)
    run(path=experiment_config, train=train, test=test, tune=tune, use_wandb=use_wandb)


if __name__ == "__main__":
    cli()

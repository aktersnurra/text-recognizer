"""Script to run experiments."""
from datetime import datetime
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Type

import click
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch import nn
from tqdm import tqdm
import wandb


SEED = 4711
CONFIGS_DIRNAME = Path(__file__).parent.resolve() / "configs"
LOGS_DIRNAME = Path(__file__).parent.resolve() / "logs"


def _create_experiment_dir(config: DictConfig) -> Path:
    """Creates log directory for experiment."""
    log_dir = (
        LOGS_DIRNAME
        / f"{config.model.type}_{config.network.type}".lower()
        / datetime.now().strftime("%m%d_%H%M%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _configure_logging(log_dir: Optional[Path], verbose: int = 0) -> None:
    """Configure the loguru logger for output to terminal and disk."""

    def _get_level(verbose: int) -> str:
        """Sets the logger level."""
        levels = {0: "WARNING", 1: "INFO", 2: "DEBUG"}
        verbose = min(verbose, 2)
        return levels[verbose]

    # Remove default logger to get tqdm to work properly.
    logger.remove()

    # Fetch verbosity level.
    level = _get_level(verbose)

    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=level)
    if log_dir is not None:
        logger.add(
            str(log_dir / "train.log"),
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def _load_config(file_path: Path) -> DictConfig:
    """Return experiment config."""
    logger.info(f"Loading config from: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Experiment config not found at: {file_path}")
    return OmegaConf.load(file_path)


def _import_class(module_and_class_name: str) -> type:
    """Import class from module."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _configure_callbacks(
    callbacks: List[DictConfig],
) -> List[Type[pl.callbacks.Callback]]:
    """Configures lightning callbacks."""
    pl_callbacks = [
        getattr(pl.callbacks, callback.type)(**callback.args) for callback in callbacks
    ]
    return pl_callbacks


def _configure_logger(
    network: Type[nn.Module], args: Dict, log_dir: Path, use_wandb: bool
) -> Type[pl.loggers.LightningLoggerBase]:
    """Configures lightning logger."""
    if use_wandb:
        pl_logger = pl.loggers.WandbLogger(save_dir=str(log_dir))
        pl_logger.watch(network)
        pl_logger.log_hyperparams(vars(args))
        return pl_logger
    return pl.loggers.TensorBoardLogger(save_dir=str(log_dir))


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


def _load_lit_model(
    lit_model_class: type, network: Type[nn.Module], config: DictConfig
) -> Type[pl.LightningModule]:
    """Load lightning model."""
    if config.load_checkpoint is not None:
        logger.info(
            f"Loading network weights from checkpoint: {config.load_checkpoint}"
        )
        return lit_model_class.load_from_checkpoint(
            config.load_checkpoint, network=network, **config.model.args
        )
    return lit_model_class(network=network, **config.model.args)


def run(
    filename: str,
    fast_dev_run: bool,
    train: bool,
    test: bool,
    tune: bool,
    use_wandb: bool,
    verbose: int = 0,
) -> None:
    """Runs experiment."""
    # Load config.
    file_path = CONFIGS_DIRNAME / filename
    config = _load_config(file_path)

    log_dir = _create_experiment_dir(config)
    _configure_logging(log_dir, verbose=verbose)
    logger.info("Starting experiment...")

    # Seed everything in the experiment.
    logger.info(f"Seeding everthing with seed={SEED}")
    pl.utilities.seed.seed_everything(SEED)

    # Load classes.
    data_module_class = _import_class(f"text_recognizer.data.{config.data.type}")
    network_class = _import_class(f"text_recognizer.networks.{config.network.type}")
    lit_model_class = _import_class(f"text_recognizer.models.{config.model.type}")

    # Initialize data object and network.
    data_module = data_module_class(**config.data.args)
    network = network_class(**data_module.config(), **config.network.args)

    # Load callback and logger.
    callbacks = _configure_callbacks(config.callbacks)
    pl_logger = _configure_logger(network, config, log_dir, use_wandb)

    # Load ligtning model.
    lit_model = _load_lit_model(lit_model_class, network, config)

    trainer = pl.Trainer(
        **config.trainer.args,
        callbacks=callbacks,
        logger=pl_logger,
        weights_save_path=str(log_dir),
    )
    if fast_dev_run:
        logger.info("Fast dev run...")
        trainer.fit(lit_model, datamodule=data_module)
    else:
        if tune:
            logger.info("Tuning learning rate and batch size...")
            trainer.tune(lit_model, datamodule=data_module)

        if train:
            logger.info("Training network...")
            trainer.fit(lit_model, datamodule=data_module)

        if test:
            logger.info("Testing network...")
            trainer.test(lit_model, datamodule=data_module)

        _save_best_weights(callbacks, use_wandb)


@click.command()
@click.option("-f", "--experiment_config", type=str, help="Path to experiment config.")
@click.option("--use_wandb", is_flag=True, help="If true, do use wandb for logging.")
@click.option("--dev", is_flag=True, help="If true, run a fast dev run.")
@click.option(
    "--tune", is_flag=True, help="If true, tune hyperparameters for training."
)
@click.option("-t", "--train", is_flag=True, help="If true, train the model.")
@click.option("-e", "--test", is_flag=True, help="If true, test the model.")
@click.option("-v", "--verbose", count=True)
def cli(
    experiment_config: str,
    use_wandb: bool,
    dev: bool,
    tune: bool,
    train: bool,
    test: bool,
    verbose: int,
) -> None:
    """Run experiment."""
    run(
        filename=experiment_config,
        fast_dev_run=dev,
        train=train,
        test=test,
        tune=tune,
        use_wandb=use_wandb,
        verbose=verbose,
    )


if __name__ == "__main__":
    cli()

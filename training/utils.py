"""Util functions for training hydra configs and pytorch lightning."""
import warnings

from omegaconf import DictConfig, OmegaConf
import loguru.logger as log
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm


@rank_zero_only
def configure_logging(level: str) -> None:
    """Configure the loguru logger for output to terminal and disk."""
    # Remove default logger to get tqdm to work properly.
    log.remove()
    log.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=level)


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

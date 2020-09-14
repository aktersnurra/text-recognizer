"""Script to run experiments."""
from datetime import datetime
from glob import glob
import importlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Tuple, Type

import click
from loguru import logger
import torch
from tqdm import tqdm
from training.gpu_manager import GPUManager
from training.trainer.callbacks import Callback, CallbackList
from training.trainer.train import Trainer
import wandb
import yaml


from text_recognizer.models import Model


EXPERIMENTS_DIRNAME = Path(__file__).parents[0].resolve() / "experiments"


DEFAULT_TRAIN_ARGS = {"batch_size": 64, "epochs": 16}


def get_level(experiment_config: Dict) -> int:
    """Sets the logger level."""
    if experiment_config["verbosity"] == 0:
        return 40
    elif experiment_config["verbosity"] == 1:
        return 20
    else:
        return 10


def create_experiment_dir(experiment_config: Dict) -> Path:
    """Create new experiment."""
    EXPERIMENTS_DIRNAME.mkdir(parents=True, exist_ok=True)
    experiment_dir = EXPERIMENTS_DIRNAME / (
        f"{experiment_config['model']}_"
        + f"{experiment_config['dataset']['type']}_"
        + f"{experiment_config['network']['type']}"
    )
    if experiment_config["resume_experiment"] is None:
        experiment = datetime.now().strftime("%m%d_%H%M%S")
        logger.debug(f"Creating a new experiment called {experiment}")
    else:
        available_experiments = glob(str(experiment_dir) + "/*")
        available_experiments.sort()
        if experiment_config["resume_experiment"] == "last":
            experiment = available_experiments[-1]
            logger.debug(f"Resuming the latest experiment {experiment}")
        else:
            experiment = experiment_config["resume_experiment"]
            if not str(experiment_dir / experiment) in available_experiments:
                raise FileNotFoundError("Experiment does not exist.")

    experiment_dir = experiment_dir / experiment

    # Create log and model directories.
    log_dir = experiment_dir / "log"
    model_dir = experiment_dir / "model"

    return experiment_dir, log_dir, model_dir


def check_args(args: Dict, train_args: Dict) -> Dict:
    """Checks that the arguments are not None."""
    args = args or {}

    # I just want to set total epochs in train args, instead of changing all parameter.
    if "epochs" in args and args["epochs"] is None:
        args["epochs"] = train_args["max_epochs"]

    # For CosineAnnealingLR.
    if "T_max" in args and args["T_max"] is None:
        args["T_max"] = train_args["max_epochs"]

    return args or {}


def load_modules_and_arguments(experiment_config: Dict) -> Tuple[Callable, Dict]:
    """Loads all modules and arguments."""
    # Import the data loader arguments.
    train_args = experiment_config.get("train_args", {})

    # Load the dataset module.
    dataset_args = experiment_config.get("dataset", {})
    dataset_args["train_args"]["batch_size"] = train_args["batch_size"]
    datasets_module = importlib.import_module("text_recognizer.datasets")
    dataset_ = getattr(datasets_module, dataset_args["type"])

    # Import the model module and model arguments.
    models_module = importlib.import_module("text_recognizer.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    # Import metrics.
    metric_fns_ = (
        {
            metric: getattr(models_module, metric)
            for metric in experiment_config["metrics"]
        }
        if experiment_config["metrics"] is not None
        else None
    )

    # Import network module and arguments.
    network_module = importlib.import_module("text_recognizer.networks")
    network_fn_ = getattr(network_module, experiment_config["network"]["type"])
    network_args = experiment_config["network"].get("args", {})

    # Criterion
    criterion_ = getattr(torch.nn, experiment_config["criterion"]["type"])
    criterion_args = experiment_config["criterion"].get("args", {})

    # Optimizers
    optimizer_ = getattr(torch.optim, experiment_config["optimizer"]["type"])
    optimizer_args = experiment_config["optimizer"].get("args", {})

    # Learning rate scheduler
    lr_scheduler_ = None
    lr_scheduler_args = None
    if experiment_config["lr_scheduler"] is not None:
        lr_scheduler_ = getattr(
            torch.optim.lr_scheduler, experiment_config["lr_scheduler"]["type"]
        )
        lr_scheduler_args = check_args(
            experiment_config["lr_scheduler"].get("args", {}), train_args
        )

    # SWA scheduler.
    if "swa_args" in experiment_config:
        swa_args = check_args(experiment_config.get("swa_args", {}), train_args)
    else:
        swa_args = None

    model_args = {
        "dataset": dataset_,
        "dataset_args": dataset_args,
        "metrics": metric_fns_,
        "network_fn": network_fn_,
        "network_args": network_args,
        "criterion": criterion_,
        "criterion_args": criterion_args,
        "optimizer": optimizer_,
        "optimizer_args": optimizer_args,
        "lr_scheduler": lr_scheduler_,
        "lr_scheduler_args": lr_scheduler_args,
        "swa_args": swa_args,
    }

    return model_class_, model_args


def configure_callbacks(experiment_config: Dict, model_dir: Dict) -> CallbackList:
    """Configure a callback list for trainer."""
    train_args = experiment_config.get("train_args", {})

    if "Checkpoint" in experiment_config["callback_args"]:
        experiment_config["callback_args"]["Checkpoint"]["checkpoint_path"] = model_dir

    # Callbacks
    callback_modules = importlib.import_module("training.trainer.callbacks")
    callbacks = [
        getattr(callback_modules, callback)(
            **check_args(experiment_config["callback_args"][callback], train_args)
        )
        for callback in experiment_config["callbacks"]
    ]

    return callbacks


def configure_logger(experiment_config: Dict, log_dir: Path) -> None:
    """Configure the loguru logger for output to terminal and disk."""
    # Have to remove default logger to get tqdm to work properly.
    logger.remove()

    # Fetch verbosity level.
    level = get_level(experiment_config)

    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=level)
    logger.add(
        str(log_dir / "train.log"),
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    )


def save_config(experiment_dir: Path, experiment_config: Dict) -> None:
    """Copy config to experiment directory."""
    config_path = experiment_dir / "config.yml"
    with open(str(config_path), "w") as f:
        yaml.dump(experiment_config, f)


def load_from_checkpoint(model: Type[Model], log_dir: Path, model_dir: Path) -> None:
    """If checkpoint exists, load model weights and optimizers from checkpoint."""
    # Get checkpoint path.
    checkpoint_path = model_dir / "last.pt"
    if checkpoint_path.exists():
        logger.info("Loading and resuming training from last checkpoint.")
        model.load_checkpoint(checkpoint_path)


def run_experiment(
    experiment_config: Dict, save_weights: bool, device: str, use_wandb: bool = False
) -> None:
    """Runs an experiment."""
    logger.info(f"Experiment config: {json.dumps(experiment_config, indent=2)}")

    # Create new experiment.
    experiment_dir, log_dir, model_dir = create_experiment_dir(experiment_config)

    # Make sure the log/model directory exists.
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load the modules and model arguments.
    model_class_, model_args = load_modules_and_arguments(experiment_config)

    # Initializes the model with experiment config.
    model = model_class_(**model_args, device=device)

    callbacks = configure_callbacks(experiment_config, model_dir)

    # Setup logger.
    configure_logger(experiment_config, log_dir)

    # Load from checkpoint if resuming an experiment.
    if experiment_config["resume_experiment"] is not None:
        load_from_checkpoint(model, log_dir, model_dir)

    logger.info(f"The class mapping is {model.mapping}")

    # Initializes Weights & Biases
    if use_wandb:
        wandb.init(project="text-recognizer", config=experiment_config)

        # Lets W&B save the model and track the gradients and optional parameters.
        wandb.watch(model.network)

    experiment_config["train_args"] = {
        **DEFAULT_TRAIN_ARGS,
        **experiment_config.get("train_args", {}),
    }

    experiment_config["experiment_group"] = experiment_config.get(
        "experiment_group", None
    )

    experiment_config["device"] = device

    # Save the config used in the experiment folder.
    save_config(experiment_dir, experiment_config)

    # Load trainer.
    trainer = Trainer(
        max_epochs=experiment_config["train_args"]["max_epochs"], callbacks=callbacks,
    )

    # Train the model.
    trainer.fit(model)

    # Run inference over test set.
    if experiment_config["test"]:
        logger.info("Loading checkpoint with the best weights.")
        model.load_from_checkpoint(model_dir / "best.pt")

        logger.info("Running inference on test set.")
        score = trainer.test(model)

        logger.info(f"Test set evaluation: {score}")

        if use_wandb:
            wandb.log(
                {
                    experiment_config["test_metric"]: score[
                        experiment_config["test_metric"]
                    ]
                }
            )

    if save_weights:
        model.save_weights(model_dir)


@click.command()
@click.argument("experiment_config",)
@click.option("--gpu", type=int, default=0, help="Provide the index of the GPU to use.")
@click.option(
    "--save",
    is_flag=True,
    help="If set, the final weights will be saved to a canonical, version-controlled location.",
)
@click.option(
    "--nowandb", is_flag=False, help="If true, do not use wandb for this run."
)
def run_cli(experiment_config: str, gpu: int, save: bool, nowandb: bool) -> None:
    """Run experiment."""
    if gpu < 0:
        gpu_manager = GPUManager(True)
        gpu = gpu_manager.get_free_gpu()
    device = "cuda:" + str(gpu)

    experiment_config = json.loads(experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
    run_experiment(experiment_config, save, device, use_wandb=not nowandb)


if __name__ == "__main__":
    run_cli()

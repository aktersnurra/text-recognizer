"""Script to run experiments."""
from datetime import datetime
from glob import glob
import importlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Dict, Tuple

import click
from loguru import logger
import torch
from tqdm import tqdm
from training.callbacks import CallbackList
from training.gpu_manager import GPUManager
from training.train import Trainer
import wandb
import yaml


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


def create_experiment_dir(model: Callable, experiment_config: Dict) -> Path:
    """Create new experiment."""
    EXPERIMENTS_DIRNAME.mkdir(parents=True, exist_ok=True)
    experiment_dir = EXPERIMENTS_DIRNAME / model.__name__
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
            logger.debug(f"Resuming the experiment {experiment}")

    experiment_dir = experiment_dir / experiment
    return experiment_dir


def load_modules_and_arguments(experiment_config: Dict) -> Tuple[Callable, Dict]:
    """Loads all modules and arguments."""
    # Import the data loader module and arguments.
    datasets_module = importlib.import_module("text_recognizer.datasets")
    data_loader_ = getattr(datasets_module, experiment_config["dataloader"])
    data_loader_args = experiment_config.get("data_loader_args", {})

    # Import the model module and model arguments.
    models_module = importlib.import_module("text_recognizer.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    # Import metrics.
    metric_fns_ = {
        metric: getattr(models_module, metric)
        for metric in experiment_config["metrics"]
    }

    # Import network module and arguments.
    network_module = importlib.import_module("text_recognizer.networks")
    network_fn_ = getattr(network_module, experiment_config["network"])
    network_args = experiment_config.get("network_args", {})

    # Criterion
    criterion_ = getattr(torch.nn, experiment_config["criterion"])
    criterion_args = experiment_config.get("criterion_args", {})

    # Optimizer
    optimizer_ = getattr(torch.optim, experiment_config["optimizer"])
    optimizer_args = experiment_config.get("optimizer_args", {})

    # Callbacks
    callback_modules = importlib.import_module("training.callbacks")
    callbacks = []
    for callback in experiment_config["callbacks"]:
        args = experiment_config["callback_args"][callback] or {}
        callbacks.append(getattr(callback_modules, callback)(**args))

    # Learning rate scheduler
    if experiment_config["lr_scheduler"] is not None:
        lr_scheduler_ = getattr(
            torch.optim.lr_scheduler, experiment_config["lr_scheduler"]
        )
        lr_scheduler_args = experiment_config.get("lr_scheduler_args", {})
    else:
        lr_scheduler_ = None
        lr_scheduler_args = None

    model_args = {
        "data_loader": data_loader_,
        "data_loader_args": data_loader_args,
        "metrics": metric_fns_,
        "network_fn": network_fn_,
        "network_args": network_args,
        "criterion": criterion_,
        "criterion_args": criterion_args,
        "optimizer": optimizer_,
        "optimizer_args": optimizer_args,
        "lr_scheduler": lr_scheduler_,
        "lr_scheduler_args": lr_scheduler_args,
    }

    return model_class_, model_args, callbacks


def run_experiment(
    experiment_config: Dict, save_weights: bool, device: str, use_wandb: bool = False
) -> None:
    """Runs an experiment."""

    # Load the modules and model arguments.
    model_class_, model_args, callbacks = load_modules_and_arguments(experiment_config)

    # Initializes the model with experiment config.
    model = model_class_(**model_args, device=device)

    # Instantiate a CallbackList.
    callbacks = CallbackList(model, callbacks)

    # Create new experiment.
    experiment_dir = create_experiment_dir(model, experiment_config)

    # Create log and model directories.
    log_dir = experiment_dir / "log"
    model_dir = experiment_dir / "model"

    # Set the model dir to be able to save checkpoints.
    model.model_dir = model_dir

    # Get checkpoint path.
    checkpoint_path = model_dir / "last.pt"
    if not checkpoint_path.exists():
        checkpoint_path = None

    # Make sure the log directory exists.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Have to remove default logger to get tqdm to work properly.
    logger.remove()

    # Fetch verbosity level.
    level = get_level(experiment_config)

    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=level)
    logger.add(
        str(log_dir / "train.log"),
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    )

    if "cuda" in device:
        gpu_index = re.sub("[^0-9]+", "", device)
        logger.info(
            f"Running experiment with config {experiment_config} on GPU {gpu_index}"
        )
    else:
        logger.info(f"Running experiment with config {experiment_config} on CPU")

    logger.info(f"The class mapping is {model.mapping}")

    # Initializes Weights & Biases
    if use_wandb:
        wandb.init(project="text-recognizer", config=experiment_config)

        # Lets W&B save the model and track the gradients and optional parameters.
        wandb.watch(model.network)

    # Pŕints a summary of the network in terminal.
    model.summary()

    experiment_config["train_args"] = {
        **DEFAULT_TRAIN_ARGS,
        **experiment_config.get("train_args", {}),
    }

    experiment_config["experiment_group"] = experiment_config.get(
        "experiment_group", None
    )

    experiment_config["device"] = device

    # Save the config used in the experiment folder.
    config_path = experiment_dir / "config.yml"
    with open(str(config_path), "w") as f:
        yaml.dump(experiment_config, f)

    trainer = Trainer(
        model=model,
        model_dir=model_dir,
        train_args=experiment_config["train_args"],
        callbacks=callbacks,
        checkpoint_path=checkpoint_path,
    )

    trainer.fit()

    logger.info("Loading checkpoint with the best weights.")
    model.load_checkpoint(model_dir / "best.pt")

    score = trainer.validate()

    logger.info(f"Validation set evaluation: {score}")

    if use_wandb:
        wandb.log({"validation_metric": score["val_accuracy"]})

    if save_weights:
        model.save_weights(model_dir)


@click.command()
@click.option(
    "--experiment_config",
    type=str,
    help='Experiment JSON, e.g. \'{"dataloader": "EmnistDataLoader", "model": "CharacterModel", "network": "mlp"}\'',
)
@click.option("--gpu", type=int, default=0, help="Provide the index of the GPU to use.")
@click.option(
    "--save",
    is_flag=True,
    help="If set, the final weights will be saved to a canonical, version-controlled location.",
)
@click.option(
    "--nowandb", is_flag=False, help="If true, do not use wandb for this run."
)
def main(experiment_config: str, gpu: int, save: bool, nowandb: bool) -> None:
    """Run experiment."""
    if gpu < 0:
        gpu_manager = GPUManager(True)
        gpu = gpu_manager.get_free_gpu()
    device = "cuda:" + str(gpu)

    experiment_config = json.loads(experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
    run_experiment(experiment_config, save, device, use_wandb=not nowandb)


if __name__ == "__main__":
    main()

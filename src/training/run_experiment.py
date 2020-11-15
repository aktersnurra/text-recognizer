"""Script to run experiments."""
from datetime import datetime
from glob import glob
import importlib
import json
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Tuple, Type
import warnings

import click
from loguru import logger
import numpy as np
import torch
from torchsummary import summary
from tqdm import tqdm
from training.gpu_manager import GPUManager
from training.trainer.callbacks import CallbackList
from training.trainer.train import Trainer
import wandb
import yaml


from text_recognizer.models import Model
from text_recognizer.networks.loss import loss as custom_loss_module

EXPERIMENTS_DIRNAME = Path(__file__).parents[0].resolve() / "experiments"


def _get_level(verbose: int) -> int:
    """Sets the logger level."""
    levels = {0: 40, 1: 20, 2: 10}
    verbose = verbose if verbose <= 2 else 2
    return levels[verbose]


def _create_experiment_dir(
    experiment_config: Dict, checkpoint: Optional[str] = None
) -> Path:
    """Create new experiment."""
    EXPERIMENTS_DIRNAME.mkdir(parents=True, exist_ok=True)
    experiment_dir = EXPERIMENTS_DIRNAME / (
        f"{experiment_config['model']}_"
        + f"{experiment_config['dataset']['type']}_"
        + f"{experiment_config['network']['type']}"
    )

    if checkpoint is None:
        experiment = datetime.now().strftime("%m%d_%H%M%S")
        logger.debug(f"Creating a new experiment called {experiment}")
    else:
        available_experiments = glob(str(experiment_dir) + "/*")
        available_experiments.sort()
        if checkpoint == "last":
            experiment = available_experiments[-1]
            logger.debug(f"Resuming the latest experiment {experiment}")
        else:
            experiment = checkpoint
            if not str(experiment_dir / experiment) in available_experiments:
                raise FileNotFoundError("Experiment does not exist.")
            logger.debug(f"Resuming the from experiment {checkpoint}")

    experiment_dir = experiment_dir / experiment

    # Create log and model directories.
    log_dir = experiment_dir / "log"
    model_dir = experiment_dir / "model"

    return experiment_dir, log_dir, model_dir


def _load_modules_and_arguments(experiment_config: Dict,) -> Tuple[Callable, Dict]:
    """Loads all modules and arguments."""
    # Load the dataset module.
    dataset_args = experiment_config.get("dataset", {})
    dataset_ = dataset_args["type"]

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
    network_fn_ = experiment_config["network"]["type"]
    network_args = experiment_config["network"].get("args", {})

    # Criterion
    if experiment_config["criterion"]["type"] in custom_loss_module.__all__:
        criterion_ = getattr(custom_loss_module, experiment_config["criterion"]["type"])
    else:
        criterion_ = getattr(torch.nn, experiment_config["criterion"]["type"])
    criterion_args = experiment_config["criterion"].get("args", {}) or {}

    # Optimizers
    optimizer_ = getattr(torch.optim, experiment_config["optimizer"]["type"])
    optimizer_args = experiment_config["optimizer"].get("args", {})

    # Learning rate scheduler
    lr_scheduler_ = None
    lr_scheduler_args = None
    if "lr_scheduler" in experiment_config:
        lr_scheduler_ = getattr(
            torch.optim.lr_scheduler, experiment_config["lr_scheduler"]["type"]
        )
        lr_scheduler_args = experiment_config["lr_scheduler"].get("args", {}) or {}

    # SWA scheduler.
    if "swa_args" in experiment_config:
        swa_args = experiment_config.get("swa_args", {}) or {}
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


def _configure_callbacks(experiment_config: Dict, model_dir: Path) -> CallbackList:
    """Configure a callback list for trainer."""
    if "Checkpoint" in experiment_config["callback_args"]:
        experiment_config["callback_args"]["Checkpoint"]["checkpoint_path"] = str(
            model_dir
        )

    # Initializes callbacks.
    callback_modules = importlib.import_module("training.trainer.callbacks")
    callbacks = []
    for callback in experiment_config["callbacks"]:
        args = experiment_config["callback_args"][callback] or {}
        callbacks.append(getattr(callback_modules, callback)(**args))

    return callbacks


def _configure_logger(log_dir: Path, verbose: int = 0) -> None:
    """Configure the loguru logger for output to terminal and disk."""
    # Have to remove default logger to get tqdm to work properly.
    logger.remove()

    # Fetch verbosity level.
    level = _get_level(verbose)

    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=level)
    logger.add(
        str(log_dir / "train.log"),
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    )


def _save_config(experiment_dir: Path, experiment_config: Dict) -> None:
    """Copy config to experiment directory."""
    config_path = experiment_dir / "config.yml"
    with open(str(config_path), "w") as f:
        yaml.dump(experiment_config, f)


def _load_from_checkpoint(
    model: Type[Model], model_dir: Path, pretrained_weights: str = None,
) -> None:
    """If checkpoint exists, load model weights and optimizers from checkpoint."""
    # Get checkpoint path.
    if pretrained_weights is not None:
        logger.info(f"Loading weights from {pretrained_weights}.")
        checkpoint_path = (
            EXPERIMENTS_DIRNAME / Path(pretrained_weights) / "model" / "best.pt"
        )
    else:
        logger.info(f"Loading weights from {model_dir}.")
        checkpoint_path = model_dir / "last.pt"
    if checkpoint_path.exists():
        logger.info("Loading and resuming training from checkpoint.")
        model.load_from_checkpoint(checkpoint_path)


def evaluate_embedding(model: Type[Model]) -> Dict:
    """Evaluates the embedding space."""
    from pytorch_metric_learning import testers
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

    accuracy_calculator = AccuracyCalculator(
        include=("mean_average_precision_at_r",), k=10
    )

    def get_all_embeddings(model: Type[Model]) -> Tuple:
        tester = testers.BaseTester()
        return tester.get_all_embeddings(model.test_dataset, model.network)

    embeddings, labels = get_all_embeddings(model)
    logger.info("Computing embedding accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        embeddings, embeddings, np.squeeze(labels), np.squeeze(labels), True
    )
    logger.info(
        f"Test set accuracy (MAP@10) = {accuracies['mean_average_precision_at_r']}"
    )
    return accuracies


def run_experiment(
    experiment_config: Dict,
    save_weights: bool,
    device: str,
    use_wandb: bool,
    train: bool,
    test: bool,
    verbose: int = 0,
    checkpoint: Optional[str] = None,
    pretrained_weights: Optional[str] = None,
) -> None:
    """Runs an experiment."""
    logger.info(f"Experiment config: {json.dumps(experiment_config)}")

    # Create new experiment.
    experiment_dir, log_dir, model_dir = _create_experiment_dir(
        experiment_config, checkpoint
    )

    # Make sure the log/model directory exists.
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load the modules and model arguments.
    model_class_, model_args = _load_modules_and_arguments(experiment_config)

    # Initializes the model with experiment config.
    model = model_class_(**model_args, device=device)

    callbacks = _configure_callbacks(experiment_config, model_dir)

    # Setup logger.
    _configure_logger(log_dir, verbose)

    # Load from checkpoint if resuming an experiment.
    resume = False
    if checkpoint is not None or pretrained_weights is not None:
        resume = True
        _load_from_checkpoint(model, model_dir, pretrained_weights)

    logger.info(f"The class mapping is {model.mapping}")

    # Initializes Weights & Biases
    if use_wandb:
        wandb.init(project="text-recognizer", config=experiment_config, resume=resume)

        # Lets W&B save the model and track the gradients and optional parameters.
        wandb.watch(model.network)

    experiment_config["experiment_group"] = experiment_config.get(
        "experiment_group", None
    )

    experiment_config["device"] = device

    # Save the config used in the experiment folder.
    _save_config(experiment_dir, experiment_config)

    # Prints a summary of the network in terminal.
    model.summary(experiment_config["train_args"]["input_shape"])

    # Load trainer.
    trainer = Trainer(
        max_epochs=experiment_config["train_args"]["max_epochs"],
        callbacks=callbacks,
        transformer_model=experiment_config["train_args"]["transformer_model"],
        max_norm=experiment_config["train_args"]["max_norm"],
    )

    # Train the model.
    if train:
        trainer.fit(model)

    # Run inference over test set.
    if test:
        logger.info("Loading checkpoint with the best weights.")
        model.load_from_checkpoint(model_dir / "best.pt")

        logger.info("Running inference on test set.")
        if experiment_config["criterion"]["type"] == "EmbeddingLoss":
            logger.info("Evaluating embedding.")
            score = evaluate_embedding(model)
        else:
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
@click.option("--test", is_flag=True, help="If true, test the model.")
@click.option("-v", "--verbose", count=True)
@click.option("--checkpoint", type=str, help="Path to the experiment.")
@click.option(
    "--pretrained_weights", type=str, help="Path to pretrained model weights."
)
@click.option(
    "--notrain", is_flag=False, help="Do not train the model.",
)
def run_cli(
    experiment_config: str,
    gpu: int,
    save: bool,
    nowandb: bool,
    notrain: bool,
    test: bool,
    verbose: int,
    checkpoint: Optional[str] = None,
    pretrained_weights: Optional[str] = None,
) -> None:
    """Run experiment."""
    if gpu < 0:
        gpu_manager = GPUManager(True)
        gpu = gpu_manager.get_free_gpu()
    device = "cuda:" + str(gpu)

    experiment_config = json.loads(experiment_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

    run_experiment(
        experiment_config,
        save,
        device,
        use_wandb=not nowandb,
        train=not notrain,
        test=test,
        verbose=verbose,
        checkpoint=checkpoint,
        pretrained_weights=pretrained_weights,
    )


if __name__ == "__main__":
    run_cli()

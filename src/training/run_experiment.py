"""Script to run experiments."""
import importlib
import os
from typing import Dict

import click
import torch
from training.train import Trainer


def run_experiment(
    experiment_config: Dict, save_weights: bool, gpu_index: int, use_wandb: bool = False
) -> None:
    """Short summary."""
    # Import the data loader module and arguments.
    datasets_module = importlib.import_module("text_recognizer.datasets")
    data_loader_ = getattr(datasets_module, experiment_config["dataloader"])
    data_loader_args = experiment_config.get("data_loader_args", {})

    # Import the model module and model arguments.
    models_module = importlib.import_module("text_recognizer.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    # Import metric.
    metric_fn_ = getattr(models_module, experiment_config["metric"])

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

    # Learning rate scheduler
    lr_scheduler_ = None
    lr_scheduler_args = None
    if experiment_config["lr_scheduler"] is not None:
        lr_scheduler_ = getattr(
            torch.optim.lr_scheduler, experiment_config["lr_scheduler"]
        )
        lr_scheduler_args = experiment_config.get("lr_scheduler_args", {})

    # Device
    # TODO fix gpu manager
    device = None

    model = model_class_(
        network_fn=network_fn_,
        network_args=network_args,
        data_loader=data_loader_,
        data_loader_args=data_loader_args,
        metrics=metric_fn_,
        criterion=criterion_,
        criterion_args=criterion_args,
        optimizer=optimizer_,
        optimizer_args=optimizer_args,
        lr_scheduler=lr_scheduler_,
        lr_scheduler_args=lr_scheduler_args,
        device=device,
    )

    # TODO: Fix checkpoint path and wandb
    trainer = Trainer(
        model=model,
        epochs=experiment_config["epochs"],
        val_metric=experiment_config["metric"],
    )

    trainer.fit()

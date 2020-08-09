"""Run a experiment from a config file."""
import json
from subprocess import run

import click
from loguru import logger
import yaml


# flake8: noqa: S404,S607,S603
def run_experiments(experiments_filename: str) -> None:
    """Run experiment from file."""
    with open(experiments_filename) as f:
        experiments_config = yaml.safe_load(f)
    num_experiments = len(experiments_config["experiments"])
    for index in range(num_experiments):
        experiment_config = experiments_config["experiments"][index]
        experiment_config["experiment_group"] = experiments_config["experiment_group"]
        cmd = f"python training/run_experiment.py --gpu=-1 --save --experiment_config='{json.dumps(experiment_config)}'"
        print(cmd)


@click.command()
@click.option(
    "--experiments_filename",
    required=True,
    type=str,
    help="Filename of Yaml file of experiments to run.",
)
def main(experiments_filename: str) -> None:
    """Parse command-line arguments and run experiments from provided file."""
    run_experiments(experiments_filename)


if __name__ == "__main__":
    main()

"""Run a experiment from a config file."""
import json

import click
from loguru import logger
import yaml


def run_experiment(experiment_filename: str) -> None:
    """Run experiment from file."""
    with open(experiment_filename) as f:
        experiments_config = yaml.safe_load(f)
    num_experiments = len(experiments_config["experiments"])
    for index in range(num_experiments):
        experiment_config = experiments_config["experiments"][index]
        experiment_config["experiment_group"] = experiments_config["experiment_group"]
        print(
            f"python training/run_experiment.py --gpu=-1 '{json.dumps(experiment_config)}'"
        )


@click.command()
@click.option(
    "--experiments_filename",
    required=True,
    type=str,
    help="Filename of Yaml file of experiments to run.",
)
def main(experiment_filename: str) -> None:
    """Parse command-line arguments and run experiments from provided file."""
    run_experiment(experiment_filename)


if __name__ == "__main__":
    main()

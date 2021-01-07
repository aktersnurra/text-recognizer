"""Run a experiment from a config file."""
import json

import click
import yaml


def run_experiments(experiments_filename: str) -> None:
    """Run experiment from file."""
    with open(experiments_filename, "r") as f:
        experiments_config = yaml.safe_load(f)

    num_experiments = len(experiments_config["experiments"])
    for index in range(num_experiments):
        experiment_config = experiments_config["experiments"][index]
        experiment_config["experiment_group"] = experiments_config["experiment_group"]
        cmd = f"poetry run run-experiment --gpu=-1 --save '{json.dumps(experiment_config)}'"
        print(cmd)


@click.command()
@click.option(
    "--experiments_filename",
    required=True,
    type=str,
    help="Filename of Yaml file of experiments to run.",
)
def run_cli(experiments_filename: str) -> None:
    """Parse command-line arguments and run experiments from provided file."""
    run_experiments(experiments_filename)


if __name__ == "__main__":
    run_cli()

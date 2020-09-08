"""W&B Sweep Functionality."""
from ast import literal_eval
import json
import os
from pathlib import Path
import signal
import subprocess  # nosec
import sys
from typing import Dict, List, Tuple

import click
import yaml

EXPERIMENTS_DIRNAME = Path(__file__).parents[0].resolve() / "experiments"


def load_config() -> Dict:
    """Load base hyperparameter config."""
    with open(str(EXPERIMENTS_DIRNAME / "default_config_emnist.yml"), "r") as f:
        default_config = yaml.safe_load(f)
    return default_config


def args_to_json(
    default_config: dict, preserve_args: tuple = ("gpu", "save")
) -> Tuple[dict, list]:
    """Convert command line arguments to nested config values.

    i.e. run_sweep.py --dataset_args.foo=1.7
    {
        "dataset_args": {
            "foo": 1.7
        }
    }

    Args:
        default_config (dict): The base config used for every experiment.
        preserve_args (tuple): Arguments preserved for all runs. Defaults to ("gpu", "save").

    Returns:
        Tuple[dict, list]: Tuple of config dictionary and list of arguments.

    """

    args = []
    config = default_config.copy()
    key, val = None, None
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=")
        elif key:
            val = arg
        else:
            key = arg
        if key and val:
            parsed_key = key.lstrip("-").split(".")
            if parsed_key[0] in preserve_args:
                args.append("--{}={}".format(parsed_key[0], val))
            else:
                nested = config
                for level in parsed_key[:-1]:
                    nested[level] = config.get(level, {})
                    nested = nested[level]
                try:
                    # Convert numerics to floats / ints
                    val = literal_eval(val)
                except ValueError:
                    pass
                nested[parsed_key[-1]] = val
            key, val = None, None
    return config, args


def main() -> None:
    """Runs a W&B sweep."""
    default_config = load_config()
    config, args = args_to_json(default_config)
    env = {
        k: v for k, v in os.environ.items() if k not in ("WANDB_PROGRAM", "WANDB_ARGS")
    }
    # pylint: disable=subprocess-popen-preexec-fn
    run = subprocess.Popen(
        ["python", "training/run_experiment.py", *args, json.dumps(config)],
        env=env,
        preexec_fn=os.setsid,
    )  # nosec
    signal.signal(signal.SIGTERM, lambda *args: run.terminate())
    run.wait()


if __name__ == "__main__":
    main()

"""Fetches model artifacts from wandb."""
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from loguru import logger as log

import wandb
from training import metadata
from wandb.apis.public import Run


def _get_run_dir(run: Run) -> Optional[Path]:
    created_at = datetime.fromisoformat(run.created_at).astimezone()
    date = created_at.date()
    hour = (created_at + created_at.utcoffset()).hour
    runs = list((metadata.RUNS_DIR / f"{date}").glob(f"{hour}-*"))
    if not runs:
        return None
    return runs[0]


def _get_best_weights(run_dir: Path) -> Optional[Path]:
    checkpoints = list(run_dir.glob("**/epoch=*.ckpt"))
    if not checkpoints:
        return None
    return checkpoints[0]


def _copy_config(run_dir: Path, dst_dir: Path) -> None:
    log.info(f"Copying config to artifact folders ({dst_dir})")
    shutil.copyfile(src=run_dir / "config.yaml", dst=dst_dir / "config.yaml")


def _copy_checkpoint(checkpoint: Path, dst_dir: Path) -> None:
    """Copy model checkpoint from local directory."""
    log.info(f"Copying best run ({checkpoint}) to artifact folders ({dst_dir})")
    shutil.copyfile(src=checkpoint, dst=dst_dir / "model.pt")


def save_model(run: Run, tag: str) -> None:
    """Save model to artifacts."""
    dst_dir = metadata.ARTIFACTS_DIR / f"{tag}_text_recognizer"
    dst_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _get_run_dir(run)
    if not run_dir:
        log.error("Could not find experiment locally!")
    best_weights = _get_best_weights(run_dir)
    if not best_weights:
        log.error("Could not find checkpoint locally!")
    _copy_config(run_dir, dst_dir)
    _copy_checkpoint(best_weights, dst_dir)
    log.info("Successfully moved model and config to artifacts directory")
    # TODO: be able to download from w&b


def find_best_run(entity: str, project: str, tag: str, metric: str, mode: str) -> Run:
    """Find the best model on wandb."""
    if mode == "min":
        default_metric_value = sys.maxsize
        sort_reverse = False
    else:
        default_metric_value = 0
        sort_reverse = True
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"tags": {"$in": [tag]}})
    runs = sorted(
        runs,
        key=lambda run: run.summary.get(metric, default_metric_value),
        reverse=sort_reverse,
    )
    best_run = runs[0]
    summary = best_run.summary
    log.info(
        (
            f"Best run is ({best_run.name}, {best_run.id}) picked from {len(runs)} "
            "runs with the following metric"
        )
    )
    log.info(
        f"{metric}: {summary[metric]}"
        # , {metric.replace('val', 'test')}: {summary[metric.replace('val', 'test')]}"
    )
    return best_run


@click.command()
@click.option("--entity", type=str, default="aktersnurra", help="Name of the author")
@click.option(
    "--project", type=str, default="text-recognizer", help="The wandb project name"
)
@click.option(
    "--tag",
    type=click.Choice(["paragraphs", "lines"]),
    default="paragraphs",
    help="Tag to filter by",
)
@click.option(
    "--metric", type=str, default="val_loss", help="Which metric to filter on"
)
@click.option(
    "--mode",
    type=click.Choice(["min", "max"]),
    default="min",
    help="Min or max value of metric",
)
def main(entity: str, project: str, tag: str, metric: str, mode: str) -> None:
    best_run = find_best_run(entity, project, tag, metric, mode)
    save_model(best_run, tag)


if __name__ == "__main__":
    main()

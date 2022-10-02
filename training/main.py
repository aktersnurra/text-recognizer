"""Loads config with hydra and runs experiment."""
import hydra
from omegaconf import DictConfig
from training.metadata import TRAINING_DIR


@hydra.main(version_base="1.2", config_path=TRAINING_DIR / "conf", config_name="config")
def main(config: DictConfig) -> None:
    """Loads config with hydra and runs the experiment."""
    import utils
    from run import run

    utils.extras(config)

    if config.get("print_config"):
        utils.print_config(config)

    return run(config)


if __name__ == "__main__":
    main()

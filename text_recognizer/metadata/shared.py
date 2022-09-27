from pathlib import Path

ESSENTIALS_FILENAME = (
    Path(__file__).parents[1].resolve() / "data" / "emnist_essentials.json"
)
DATA_DIRNAME = Path(__file__).resolve().parents[2] / "data"
DOWNLOADED_DATA_DIRNAME = DATA_DIRNAME / "downloded"

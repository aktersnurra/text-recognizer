"""Module for creating EMNIST test support files."""
from pathlib import Path
import shutil

from text_recognizer.datasets.emnist_dataset import (
    fetch_emnist_dataset,
    load_emnist_mapping,
)
from text_recognizer.util import write_image

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "emnist"


def create_emnist_support_files() -> None:
    """Create support images for test of CharacterPredictor class."""
    shutil.rmtree(SUPPORT_DIRNAME, ignore_errors=True)
    SUPPORT_DIRNAME.mkdir()

    dataset = fetch_emnist_dataset(split="byclass", train=False)
    mapping = load_emnist_mapping()

    for index in [5, 7, 9]:
        image, label = dataset[index]
        if len(image.shape) == 3:
            image = image.squeeze(0)
        image = image.numpy()
        label = mapping[int(label)]
        print(index, label)
        write_image(image, str(SUPPORT_DIRNAME / f"{label}.png"))


if __name__ == "__main__":
    create_emnist_support_files()

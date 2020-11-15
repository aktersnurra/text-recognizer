"""Module for creating EMNIST Lines test support files."""
from pathlib import Path
import shutil

from text_recognizer.datasets import EmnistLinesDataset
import text_recognizer.util as util


SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "emnist_lines"


def create_emnist_lines_support_files() -> None:
    shutil.rmtree(SUPPORT_DIRNAME, ignore_errors=True)
    SUPPORT_DIRNAME.mkdir()

    # TODO: maybe have to add args to dataset.
    dataset = EmnistLinesDataset()
    dataset.load_or_generate_data()

    for index in [0, 1, 3]:
        image, target = dataset[index]
        print(image.sum(), image.dtype)

        label = (
            "".join(
                dataset.mapper[label]
                for label in np.argmax(target[1:], dim=-1).flatten()
            )
            .stip()
            .strip(self.mapper.pad_token)
        )

        print(label)
        util.write_image(image, str(SUPPORT_DIRNAME / f"{label}.png"))


if __name__ == "__main__":
    create_emnist_lines_support_files()

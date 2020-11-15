"""Module for creating IAM Lines test support files."""
# flake8: noqa
from pathlib import Path
import shutil

import numpy as np

from text_recognizer.datasets import IamLinesDataset
import text_recognizer.util as util


SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "iam_lines"


def create_emnist_lines_support_files() -> None:
    """Create IAM Lines test images."""
    shutil.rmtree(SUPPORT_DIRNAME, ignore_errors=True)
    SUPPORT_DIRNAME.mkdir()

    # TODO: maybe have to add args to dataset.
    dataset = IamLinesDataset(
        init_token="<sos>",
        pad_token="_",
        eos_token="<eos>",
        transform=[{"type": "ToTensor", "args": {}}],
        target_transform=[
            {
                "type": "AddTokens",
                "args": {"init_token": "<sos>", "pad_token": "_", "eos_token": "<eos>"},
            }
        ],
    )
    dataset.load_or_generate_data()

    for index in [0, 1, 3]:
        image, target = dataset[index]
        if len(image.shape) == 3:
            image = image.squeeze(0)
        print(image.sum(), image.dtype)

        label = "".join(dataset.mapper(label) for label in target[1:]).strip(
            dataset.mapper.pad_token
        )
        print(label)
        image = image.numpy()
        util.write_image(image, str(SUPPORT_DIRNAME / f"{label}.png"))


if __name__ == "__main__":
    create_emnist_lines_support_files()

"""Base lightning DataModule class."""
from pathlib import Path
from typing import Dict, Tuple, Type

import attr
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from text_recognizer.data.base_mapping import AbstractMapping
from text_recognizer.data.base_dataset import BaseDataset


def load_and_print_info(data_module_class: type) -> None:
    """Load dataset and print dataset information."""
    dataset = data_module_class()
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


@attr.s(repr=False)
class BaseDataModule(LightningDataModule):
    """Base PyTorch Lightning DataModule."""

    def __attrs_pre_init__(self) -> None:
        super().__init__()

    mapping: Type[AbstractMapping] = attr.ib()
    batch_size: int = attr.ib(default=16)
    num_workers: int = attr.ib(default=0)
    pin_memory: bool = attr.ib(default=True)

    # Placeholders
    data_train: BaseDataset = attr.ib(init=False, default=None)
    data_val: BaseDataset = attr.ib(init=False, default=None)
    data_test: BaseDataset = attr.ib(init=False, default=None)
    dims: Tuple[int, ...] = attr.ib(init=False, default=None)
    output_dims: Tuple[int, ...] = attr.ib(init=False, default=None)

    @classmethod
    def data_dirname(cls) -> Path:
        """Return the path to the base data directory."""
        return Path(__file__).resolve().parents[2] / "data"

    def config(self) -> Dict:
        """Return important settings of the dataset."""
        return {
            "input_dim": self.dims,
            "output_dims": self.output_dims,
        }

    def prepare_data(self) -> None:
        """Prepare data for training."""
        pass

    def setup(self, stage: str = None) -> None:
        """Split into train, val, test, and set dims.

        Should assign `torch Dataset` objects to self.data_train, self.data_val, and
            optionally self.data_test.

        Args:
            stage (Any): Variable to set splits.

        """
        pass

    def train_dataloader(self) -> DataLoader:
        """Retun DataLoader for train data."""
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for val data."""
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for val data."""
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

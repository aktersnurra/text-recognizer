"""Base lightning DataModule class."""
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader


def load_and_print_info(data_module_class: type) -> None:
    """Load dataset and print dataset information."""
    dataset = data_module_class()
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


class BaseDataModule(pl.LightningDataModule):
    """Base PyTorch Lightning DataModule."""

    def __init__(self, batch_size: int = 128, num_workers: int = 0) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Placeholders for subclasses.
        self.dims = None
        self.output_dims = None
        self.mapping = None

    @classmethod
    def data_dirname(cls) -> Path:
        """Return the path to the base data directory."""
        return Path(__file__).resolve().parents[2] / "data"

    def config(self) -> Dict:
        """Return important settings of the dataset."""
        return {
            "input_dim": self.dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping,
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
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def train_dataloader(self) -> DataLoader:
        """Retun DataLoader for train data."""
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for val data."""
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for val data."""
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

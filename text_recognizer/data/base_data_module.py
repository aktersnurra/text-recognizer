"""Base lightning DataModule class."""
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from text_recognizer.data.base_dataset import BaseDataset
from text_recognizer.data.mappings import AbstractMapping

T = TypeVar("T")


def load_and_print_info(data_module_class: type) -> None:
    """Load dataset and print dataset information."""
    dataset = data_module_class()
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


class BaseDataModule(LightningDataModule):
    """Base PyTorch Lightning DataModule."""

    def __init__(
        self,
        mapping: Type[AbstractMapping],
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_fraction: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.transform = transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Placeholders
        self.data_train: BaseDataset
        self.data_val: BaseDataset
        self.data_test: BaseDataset
        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]

    @classmethod
    def data_dirname(cls: T) -> Path:
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

    def setup(self, stage: Optional[str] = None) -> None:
        """Split into train, val, test, and set dims.

        Should assign `torch Dataset` objects to self.data_train, self.data_val, and
            optionally self.data_test.

        Args:
            stage (Optional[str]): Variable to set splits.

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

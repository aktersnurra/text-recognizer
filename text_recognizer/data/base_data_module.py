"""Base lightning DataModule class."""
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar

from attrs import define, field
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from text_recognizer.data.base_dataset import BaseDataset
from text_recognizer.data.mappings.base import AbstractMapping

T = TypeVar("T")


def load_and_print_info(data_module_class: type) -> None:
    """Load dataset and print dataset information."""
    dataset = data_module_class()
    dataset.prepare_data()
    dataset.setup()
    print(dataset)


@define(repr=False)
class BaseDataModule(LightningDataModule):
    """Base PyTorch Lightning DataModule."""

    def __attrs_post_init__(self) -> None:
        """Pre init constructor."""
        super().__init__()

    mapping: Type[AbstractMapping] = field()
    transform: Optional[Callable] = field(default=None)
    test_transform: Optional[Callable] = field(default=None)
    target_transform: Optional[Callable] = field(default=None)
    train_fraction: float = field(default=0.8)
    batch_size: int = field(default=16)
    num_workers: int = field(default=0)
    pin_memory: bool = field(default=True)

    # Placeholders
    data_train: BaseDataset = field(init=False, default=None)
    data_val: BaseDataset = field(init=False, default=None)
    data_test: BaseDataset = field(init=False, default=None)
    dims: Tuple[int, ...] = field(init=False, default=None)
    output_dims: Tuple[int, ...] = field(init=False, default=None)

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

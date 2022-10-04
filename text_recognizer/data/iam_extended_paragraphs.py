"""IAM original and sythetic dataset class."""
from typing import Callable, Optional

from torch.utils.data import ConcatDataset

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.iam_paragraphs import IAMParagraphs
from text_recognizer.data.iam_synthetic_paragraphs import IAMSyntheticParagraphs
from text_recognizer.data.transforms.pad import Pad
from text_recognizer.data.tokenizer import Tokenizer
from text_recognizer.data.transforms.paragraph import ParagraphStem
import text_recognizer.metadata.iam_paragraphs as metadata


class IAMExtendedParagraphs(BaseDataModule):
    """A dataset with synthetic and real handwritten paragraph."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_fraction: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__(
            tokenizer,
            transform,
            test_transform,
            target_transform,
            train_fraction,
            batch_size,
            num_workers,
            pin_memory,
        )
        self.iam_paragraphs = IAMParagraphs(
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            train_fraction=self.train_fraction,
            transform=self.transform,
            test_transform=self.test_transform,
            target_transform=self.target_transform,
        )
        self.iam_synthetic_paragraphs = IAMSyntheticParagraphs(
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            train_fraction=self.train_fraction,
            transform=self.transform,
            test_transform=self.test_transform,
            target_transform=self.target_transform,
        )

        self.dims = self.iam_paragraphs.dims
        self.output_dims = self.iam_paragraphs.output_dims

    def prepare_data(self) -> None:
        """Prepares the paragraphs data."""
        self.iam_paragraphs.prepare_data()
        self.iam_synthetic_paragraphs.prepare_data()

    def setup(self, stage: str = None) -> None:
        """Loads data for training/testing."""
        self.iam_paragraphs.setup(stage)
        self.iam_synthetic_paragraphs.setup(stage)

        self.data_train = ConcatDataset(
            [self.iam_paragraphs.data_train, self.iam_synthetic_paragraphs.data_train]
        )
        self.data_val = self.iam_paragraphs.data_val
        self.data_test = self.iam_paragraphs.data_test

    def __repr__(self) -> str:
        """Returns info about the dataset."""
        basic = (
            "IAM Original and Synthetic Paragraphs Dataset\n"  # pylint: disable=no-member
            f"Num classes: {len(self.tokenizer)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        x = x[0] if isinstance(x, list) else x
        xt = xt[0] if isinstance(xt, list) else xt
        data = (
            "Train/val/test sizes: "
            f"{len(self.data_train)}, "
            f"{len(self.data_val)}, "
            f"{len(self.data_test)}\n"
            "Train Batch x stats: "
            f"{(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: "
            f"{(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def show_dataset_info() -> None:
    """Displays Iam extended dataset information."""
    transform = ParagraphStem(augment=False)
    test_transform = ParagraphStem(augment=False)
    target_transform = Pad(metadata.MAX_LABEL_LENGTH, 3)
    load_and_print_info(
        IAMExtendedParagraphs(
            transform=transform,
            test_transform=test_transform,
            target_transform=target_transform,
        )
    )

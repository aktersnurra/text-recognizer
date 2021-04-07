"""IAM original and sythetic dataset class."""
from torch.utils.data import ConcatDataset

from text_recognizer.data.base_dataset import BaseDataset
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.iam_paragraphs import IAMParagraphs
from text_recognizer.data.iam_synthetic_paragraphs import IAMSyntheticParagraphs


class IAMExtendedParagraphs(BaseDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 0,
        train_fraction: float = 0.8,
        augment: bool = True,
    ) -> None:
        super().__init__(batch_size, num_workers)

        self.iam_paragraphs = IAMParagraphs(
            batch_size, num_workers, train_fraction, augment,
        )
        self.iam_synthetic_paragraphs = IAMSyntheticParagraphs(
            batch_size, num_workers, train_fraction, augment,
        )

        self.dims = self.iam_paragraphs.dims
        self.output_dims = self.iam_paragraphs.output_dims
        self.mapping = self.iam_paragraphs.mapping
        self.inverse_mapping = self.iam_paragraphs.inverse_mapping

        self.data_train: BaseDataset = None
        self.data_val: BaseDataset = None
        self.data_test: BaseDataset = None

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
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def show_dataset_info() -> None:
    load_and_print_info(IAMExtendedParagraphs)

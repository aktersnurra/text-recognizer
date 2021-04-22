"""Dataset modules."""
from .base_dataset import BaseDataset, convert_strings_to_labels, split_dataset
from .base_data_module import BaseDataModule, load_and_print_info
from .download_utils import download_dataset
from .iam_paragraphs import IAMParagraphs
from .iam_synthetic_paragraphs import IAMSyntheticParagraphs
from .iam_extended_paragraphs import IAMExtendedParagraphs

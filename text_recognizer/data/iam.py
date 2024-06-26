"""Class for loading the IAM dataset.

Which encompasses both paragraphs and lines, with associated utilities.
"""

import os
import xml.etree.ElementTree as ElementTree
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import toml
from boltons.cacheutils import cachedproperty
from loguru import logger as log

import text_recognizer.metadata.iam as metadata
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.utils.download_utils import download_dataset


class IAM(BaseDataModule):
    r"""The IAM Lines dataset.

    First published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray
    levels. From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text
    lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has
        contributed to one set only.
    """

    def __init__(self) -> None:
        super().__init__()

        self.metadata: Dict = toml.load(metadata.METADATA_FILENAME)

    def prepare_data(self) -> None:
        """Prepares the IAM dataset."""
        if self.xml_filenames:
            return
        filename = download_dataset(self.metadata, metadata.DL_DATA_DIRNAME)
        _extract_raw_dataset(filename, metadata.DL_DATA_DIRNAME)

    @property
    def xml_filenames(self) -> List[Path]:
        """Returns the xml filenames."""
        return list((metadata.EXTRACTED_DATASET_DIRNAME / "xml").glob("*.xml"))

    @property
    def form_filenames(self) -> List[Path]:
        """Returns the form filenames."""
        return list((metadata.EXTRACTED_DATASET_DIRNAME / "forms").glob("*.jpg"))

    @property
    def form_filenames_by_id(self) -> Dict[str, Path]:
        """Returns dictionary with filename and path."""
        return {filename.stem: filename for filename in self.form_filenames}

    @property
    def split_by_id(self) -> Dict[str, str]:
        """Splits files into train and test."""
        return {
            filename.stem: "test"
            if filename.stem in self.metadata["test_ids"]
            else "train"
            for filename in self.form_filenames
        }

    @cachedproperty
    def line_strings_by_id(self) -> Dict[str, List[str]]:
        """Return a dict from name of IAM form to list of line texts in it."""
        return {
            filename.stem: _get_line_strings_from_xml_file(filename)
            for filename in self.xml_filenames
        }

    @cachedproperty
    def line_regions_by_id(self) -> Dict[str, List[Dict[str, int]]]:
        """Return a dict from name IAM form to list of (x1, x2, y1, y2)."""
        return {
            filename.stem: _get_line_regions_from_xml_file(filename)
            for filename in self.xml_filenames
        }

    def __repr__(self) -> str:
        """Return info about the dataset."""
        return (
            "IAM Dataset\n"
            f"Num forms total: {len(self.xml_filenames)}\n"
            f"Num in test set: {len(self.metadata['test_ids'])}\n"
        )


def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    log.info("Extracting IAM data...")
    curdir = os.getcwd()
    os.chdir(dirname)
    with zipfile.ZipFile(filename, "r") as f:
        f.extractall()
    os.chdir(curdir)


def _get_line_strings_from_xml_file(filename: str) -> List[str]:
    """Get the text content of each line. Note that we replace &quot: with "."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [el.attrib["text"].replace("&quot;", '"') for el in xml_line_elements]


def _get_line_regions_from_xml_file(filename: str) -> List[Dict[str, int]]:
    """Get line region dict for each line."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [_get_line_region_from_xml_file(el) for el in xml_line_elements]


def _get_line_region_from_xml_file(xml_line: Any) -> Dict[str, int]:
    word_elements = xml_line.findall("word/cmp")
    x1s = [int(el.attrib["x"]) for el in word_elements]
    y1s = [int(el.attrib["y"]) for el in word_elements]
    x2s = [int(el.attrib["x"]) + int(el.attrib["width"]) for el in word_elements]
    y2s = [int(el.attrib["y"]) + int(el.attrib["height"]) for el in word_elements]
    return {
        "x1": min(x1s) // metadata.DOWNSAMPLE_FACTOR - metadata.LINE_REGION_PADDING,
        "y1": min(y1s) // metadata.DOWNSAMPLE_FACTOR - metadata.LINE_REGION_PADDING,
        "x2": max(x2s) // metadata.DOWNSAMPLE_FACTOR + metadata.LINE_REGION_PADDING,
        "y2": max(y2s) // metadata.DOWNSAMPLE_FACTOR + metadata.LINE_REGION_PADDING,
    }


def download_iam() -> None:
    """Downloads and prints IAM dataset."""
    load_and_print_info(IAM)

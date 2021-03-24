"""Preprocessor for extracting word letters from the IAM dataset.

The code is mostly stolen from:

    https://github.com/facebookresearch/gtn_applications/blob/master/datasets/iamdb.py

"""

import collections
import itertools
from pathlib import Path
import re
from typing import List, Optional, Union

import click
from loguru import logger
import torch


def load_metadata(
    data_dir: Path, wordsep: str, use_words: bool = False
) -> collections.defaultdict:
    """Loads IAM metadata and returns it as a dictionary."""
    forms = collections.defaultdict(list)
    filename = "words.txt" if use_words else "lines.txt"

    with open(data_dir / "ascii" / filename, "r") as f:
        lines = (line.strip().split() for line in f if line[0] != "#")
        for line in lines:
            # Skip word segmentation errors.
            if use_words and line[1] == "err":
                continue
            text = " ".join(line[8:])

            # Remove garbage tokens:
            text = text.replace("#", "")

            # Swap word sep form | to wordsep
            text = re.sub(r"\|+|\s", wordsep, text).strip(wordsep)
            form_key = "-".join(line[0].split("-")[:2])
            line_key = "-".join(line[0].split("-")[:3])
            box_idx = 4 - use_words
            box = tuple(int(val) for val in line[box_idx : box_idx + 4])
            forms[form_key].append({"key": line_key, "box": box, "text": text})
    return forms


class Preprocessor:
    """A preprocessor for the IAM dataset."""

    # TODO: add lower case only to when generating...

    def __init__(
        self,
        data_dir: Union[str, Path],
        num_features: int,
        tokens_path: Optional[Union[str, Path]] = None,
        lexicon_path: Optional[Union[str, Path]] = None,
        use_words: bool = False,
        prepend_wordsep: bool = False,
    ) -> None:
        self.wordsep = "▁"
        self._use_word = use_words
        self._prepend_wordsep = prepend_wordsep

        self.data_dir = Path(data_dir)

        self.forms = load_metadata(self.data_dir, self.wordsep, use_words=use_words)

        # Load the set of graphemes:
        graphemes = set()
        for _, form in self.forms.items():
            for line in form:
                graphemes.update(line["text"].lower())
        self.graphemes = sorted(graphemes)

        # Build the token-to-index and index-to-token maps.
        if tokens_path is not None:
            with open(tokens_path, "r") as f:
                self.tokens = [line.strip() for line in f]
        else:
            self.tokens = self.graphemes

        if lexicon_path is not None:
            with open(lexicon_path, "r") as f:
                lexicon = (line.strip().split() for line in f)
                lexicon = {line[0]: line[1:] for line in lexicon}
                self.lexicon = lexicon
        else:
            self.lexicon = None

        self.graphemes_to_index = {t: i for i, t in enumerate(self.graphemes)}
        self.tokens_to_index = {t: i for i, t in enumerate(self.tokens)}
        self.num_features = num_features
        self.text = []

    @property
    def num_tokens(self) -> int:
        """Returns the number or tokens."""
        return len(self.tokens)

    @property
    def use_words(self) -> bool:
        """If words are used."""
        return self._use_word

    def extract_train_text(self) -> None:
        """Extracts training text."""
        keys = []
        with open(self.data_dir / "task" / "trainset.txt") as f:
            keys.extend((line.strip() for line in f))

        for _, examples in self.forms.items():
            for example in examples:
                if example["key"] not in keys:
                    continue
                self.text.append(example["text"].lower())

    def to_index(self, line: str) -> torch.LongTensor:
        """Converts text to a tensor of indices."""
        token_to_index = self.graphemes_to_index
        if self.lexicon is not None:
            if len(line) > 0:
                # If the word is not found in the lexicon, fall back to letters.
                line = [
                    t
                    for w in line.split(self.wordsep)
                    for t in self.lexicon.get(w, self.wordsep + w)
                ]
            token_to_index = self.tokens_to_index
        if self._prepend_wordsep:
            line = itertools.chain([self.wordsep], line)
        return torch.LongTensor([token_to_index[t] for t in line])

    def to_text(self, indices: List[int]) -> str:
        """Converts indices to text."""
        # Roughly the inverse of `to_index`
        encoding = self.graphemes
        if self.lexicon is not None:
            encoding = self.tokens
        return self._post_process(encoding[i] for i in indices)

    def tokens_to_text(self, indices: List[int]) -> str:
        """Converts tokens to text."""
        return self._post_process(self.tokens[i] for i in indices)

    def _post_process(self, indices: List[int]) -> str:
        """A list join."""
        return "".join(indices).strip(self.wordsep)


@click.command()
@click.option("--data_dir", type=str, default=None, help="Path to iam dataset")
@click.option(
    "--use_words", is_flag=True, help="Load word segmented dataset instead of lines"
)
@click.option(
    "--save_text", type=str, default=None, help="Path to save parsed train text"
)
@click.option("--save_tokens", type=str, default=None, help="Path to save tokens")
def cli(
    data_dir: Optional[str],
    use_words: bool,
    save_text: Optional[str],
    save_tokens: Optional[str],
) -> None:
    """CLI for extracting text data from the iam dataset."""
    if data_dir is None:
        data_dir = (
            Path(__file__).resolve().parents[3] / "data" / "raw" / "iam" / "iamdb"
        )
        logger.debug(f"Using data dir: {data_dir}")
        if not data_dir.exists():
            raise RuntimeError(f"Could not locate iamdb directory at {data_dir}")
    else:
        data_dir = Path(data_dir)

    preprocessor = Preprocessor(data_dir, 64, use_words=use_words)
    preprocessor.extract_train_text()

    processed_dir = data_dir.parents[2] / "processed" / "iam_lines"
    logger.debug(f"Saving processed files at: {processed_dir}")

    if save_text is not None:
        logger.info("Saving training text")
        with open(processed_dir / save_text, "w") as f:
            f.write("\n".join(t for t in preprocessor.text))

    if save_tokens is not None:
        logger.info("Saving tokens")
        with open(processed_dir / save_tokens, "w") as f:
            f.write("\n".join(preprocessor.tokens))


if __name__ == "__main__":
    cli()

"""Downloading the Brown corpus with NLTK for sentence generating."""

import itertools
import re
import string
from typing import Optional

import nltk
from nltk.corpus.reader.util import ConcatenatedCorpusView
import numpy as np

from text_recognizer.datasets.util import DATA_DIRNAME

NLTK_DATA_DIRNAME = DATA_DIRNAME / "raw" / "nltk"


class SentenceGenerator:
    """Generates text sentences using the Brown corpus."""

    def __init__(self, max_length: Optional[int] = None) -> None:
        """Loads the corpus and sets word start indices."""
        self.corpus = brown_corpus()
        self.word_start_indices = [0] + [
            _.start(0) + 1 for _ in re.finditer(" ", self.corpus)
        ]
        self.max_length = max_length

    def generate(self, max_length: Optional[int] = None) -> str:
        """Generates a word or sentences from the Brown corpus.

        Sample a string from the Brown corpus of length at least one word and at most max_length, padding to
        max_length with the '_' characters if sentence is shorter.

        Args:
            max_length (Optional[int]): The maximum number of characters in the sentence. Defaults to None.

        Returns:
            str: A sentence from the Brown corpus.

        Raises:
            ValueError: If max_length was not specified at initialization and not given as an argument.

        """
        if max_length is None:
            max_length = self.max_length
        if max_length is None:
            raise ValueError(
                "Must provide max_length to this method or when making this object."
            )

        index = np.random.randint(0, len(self.word_start_indices) - 1)
        start_index = self.word_start_indices[index]
        end_index_candidates = []
        for index in range(index + 1, len(self.word_start_indices)):
            if self.word_start_indices[index] - start_index > max_length:
                break
            end_index_candidates.append(self.word_start_indices[index])
        end_index = np.random.choice(end_index_candidates)
        sampled_text = self.corpus[start_index:end_index].strip()
        padding = "_" * (max_length - len(sampled_text))
        return sampled_text + padding


def brown_corpus() -> str:
    """Returns a single string with the Brown corpus with all punctuations stripped."""
    sentences = load_nltk_brown_corpus()
    corpus = " ".join(itertools.chain.from_iterable(sentences))
    corpus = corpus.translate({ord(c): None for c in string.punctuation})
    corpus = re.sub(" +", " ", corpus)
    return corpus


def load_nltk_brown_corpus() -> ConcatenatedCorpusView:
    """Load the Brown corpus using the NLTK library."""
    nltk.data.path.append(NLTK_DATA_DIRNAME)
    try:
        nltk.corpus.brown.sents()
    except LookupError:
        NLTK_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        nltk.download("brown", download_dir=NLTK_DATA_DIRNAME)
    return nltk.corpus.brown.sents()

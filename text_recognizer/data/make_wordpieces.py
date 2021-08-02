"""Creates word pieces from a text file.

Most code stolen from:

    https://github.com/facebookresearch/gtn_applications/blob/master/scripts/make_wordpieces.py

"""
import io
from pathlib import Path
from typing import List, Optional, Union

import click
from loguru import logger as log
import sentencepiece as spm

from text_recognizer.data.iam_preprocessor import load_metadata


def iamdb_pieces(
    data_dir: Path, text_file: str, num_pieces: int, output_prefix: str
) -> None:
    """Creates word pieces from the iamdb train text."""
    # Load training text.
    with open(data_dir / text_file, "r") as f:
        text = [line.strip() for line in f]

    sp = train_spm_model(
        iter(text),
        num_pieces + 1,  # To account for <unk>
        user_symbols=["/"],  # added so token is in the output set
    )

    vocab = sorted(set(w for t in text for w in t.split("▁") if w))
    if "move" not in vocab:
        raise RuntimeError("`MOVE` not in vocab")

    save_pieces(sp, num_pieces, data_dir, output_prefix, vocab)


def train_spm_model(
    sentences: iter, vocab_size: int, user_symbols: Union[str, List[str]] = ""
) -> spm.SentencePieceProcessor:
    """Trains the sentence piece model."""
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=sentences,
        model_writer=model,
        vocab_size=vocab_size,
        bos_id=-1,
        eos_id=-1,
        character_coverage=1.0,
        user_defined_symbols=user_symbols,
    )
    sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
    return sp


def save_pieces(
    sp: spm.SentencePieceProcessor,
    num_pieces: int,
    data_dir: Path,
    output_prefix: str,
    vocab: set,
) -> None:
    """Saves word pieces to disk."""
    log.info(f"Generating word piece list of size {num_pieces}.")
    pieces = [sp.id_to_piece(i) for i in range(1, num_pieces + 1)]
    log.info(f"Encoding vocabulary of size {len(vocab)}.")
    encoded_vocab = [sp.encode_as_pieces(v) for v in vocab]

    # Save pieces to file.
    with open(data_dir / f"{output_prefix}_tokens_{num_pieces}.txt", "w") as f:
        f.write("\n".join(pieces))

    # Save lexicon to a file.
    with open(data_dir / f"{output_prefix}_lex_{num_pieces}.txt", "w") as f:
        for v, p in zip(vocab, encoded_vocab):
            f.write(f"{v} {' '.join(p)}\n")


@click.command()
@click.option("--data_dir", type=str, default=None, help="Path to processed iam dir.")
@click.option(
    "--text_file", type=str, default=None, help="Name of sentence piece training text."
)
@click.option(
    "--output_prefix",
    type=str,
    default="word_pieces",
    help="Prefix name to store tokens and lexicon.",
)
@click.option("--num_pieces", type=int, default=1000, help="Number of word pieces.")
def cli(
    data_dir: Optional[str],
    text_file: Optional[str],
    output_prefix: Optional[str],
    num_pieces: Optional[int],
) -> None:
    """CLI for training the sentence piece model."""
    if data_dir is None:
        data_dir = (
            Path(__file__).resolve().parents[2] / "data" / "processed" / "iam_lines"
        )
        log.debug(f"Using data dir: {data_dir}")
        if not data_dir.exists():
            raise RuntimeError(f"Could not locate iamdb directory at {data_dir}")
    else:
        data_dir = Path(data_dir)

    iamdb_pieces(data_dir, text_file, num_pieces, output_prefix)


if __name__ == "__main__":
    cli()

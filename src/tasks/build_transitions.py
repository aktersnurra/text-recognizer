"""Builds transition graph.

Most code stolen from here:

    https://github.com/facebookresearch/gtn_applications/blob/master/scripts/build_transitions.py

"""

import collections
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import gtn
from loguru import logger


START_IDX = -1
END_IDX = -2
WORDSEP = "▁"


def build_graph(ngrams: List, disable_backoff: bool = False) -> gtn.Graph:
    """Returns a gtn Graph based on the ngrams."""
    graph = gtn.Graph(False)
    ngram = len(ngrams)
    state_to_node = {}

    def get_node(state: Optional[List]) -> Any:
        node = state_to_node.get(state, None)

        if node is not None:
            return node

        start = state == tuple([START_IDX]) if ngram > 1 else True
        end = state == tuple([END_IDX]) if ngram > 1 else True
        node = graph.add_node(start, end)
        state_to_node[state] = node

        if not disable_backoff and not end:
            # Add back off when adding node.
            for n in range(1, len(state) + 1):
                backoff_node = state_to_node.get(state[n:], None)

                # Epsilon transition to the back-off state.
                if backoff_node is not None:
                    graph.add_arc(node, backoff_node, gtn.epsilon)
                    break
        return node

    for grams in ngrams:
        for gram in grams:
            istate, ostate = gram[:-1], gram[len(gram) - ngram + 1 :]
            inode = get_node(istate)

            if END_IDX not in gram[1:] and gram[1:] not in state_to_node:
                raise ValueError(
                    "Ill formed counts: if (x, y_1, ..., y_{n-1}) is above"
                    "the n-gram threshold, then (y_1, ..., y_{n-1}) must be"
                    "above the (n-1)-gram threshold"
                )

            if END_IDX in ostate:
                # Merge all state having </s> into one as final graph generated
                # will be similar.
                ostate = tuple([END_IDX])

            onode = get_node(ostate)
            # p(gram[-1] | gram[:-1])
            graph.add_arc(
                inode, onode, gtn.epsilon if gram[-1] == END_IDX else gram[-1]
            )
    return graph


def count_ngrams(lines: List, ngram: List, tokens_to_index: Dict) -> List:
    """Counts the number of ngrams."""
    counts = [collections.Counter() for _ in range(ngram)]
    for line in lines:
        # Prepend implicit start token.
        token_line = [START_IDX]
        for t in line:
            token_line.append(tokens_to_index[t])
        token_line.append(END_IDX)
        for n, counter in enumerate(counts):
            start_offset = n == 0
            end_offset = ngram == 1
            for e in range(n + start_offset, len(token_line) - end_offset):
                counter[tuple(token_line[e - n : e + 1])] += 1

    return counts


def prune_ngrams(ngrams: List, prune: List) -> List:
    """Prunes ngrams."""
    pruned_ngrams = []
    for n, grams in enumerate(ngrams):
        grams = grams.most_common()
        pruned_grams = [gram for gram, c in grams if c > prune[n]]
        pruned_ngrams.append(pruned_grams)
    return pruned_ngrams


def add_blank_grams(pruned_ngrams: List, num_tokens: int, blank: str) -> List:
    """Adds blank token to grams."""
    all_grams = [gram for grams in pruned_ngrams for gram in grams]
    maxorder = len(pruned_ngrams)
    blank_grams = {}
    if blank == "forced":
        pruned_ngrams = [pruned_ngrams[0] if i == 0 else [] for i in range(maxorder)]
    pruned_ngrams[0].append(tuple([num_tokens]))
    blank_grams[tuple([num_tokens])] = True

    for gram in all_grams:
        # Iterate over all possibilities by using a vector of 0s, 1s to
        # denote whether a blank is being used at each position.
        if blank == "optional":
            # Given a gram ab.. if order n, we have n + 1 positions
            # available whether to use blank or not.
            onehot_vectors = itertools.product([0, 1], repeat=len(gram) + 1)
        elif blank == "forced":
            # Must include a blank token in between.
            onehot_vectors = [[1] * (len(gram) + 1)]
        else:
            raise ValueError(
                "Invalid value specificed for blank. Must be in |optional|forced|none|"
            )

    for j in onehot_vectors:
        new_array = []
        for idx, oz in enumerate(j[:-1]):
            if oz == 1 and gram[idx] != START_IDX:
                new_array.append(num_tokens)
            new_array.append(gram[idx])
        if j[-1] == 1 and gram[-1] != END_IDX:
            new_array.append(num_tokens)
        for n in range(maxorder):
            for e in range(n, len(new_array)):
                cur_gram = tuple(new_array[e - n : e + 1])
                if num_tokens in cur_gram and cur_gram not in blank_grams:
                    pruned_ngrams[n].append(cur_gram)
                    blank_grams[cur_gram] = True

    return pruned_ngrams


def add_self_loops(pruned_ngrams: List) -> List:
    """Adds self loops to the ngrams."""
    maxorder = len(pruned_ngrams)

    # Use dict for fast search.
    all_grams = set([gram for grams in pruned_ngrams for gram in grams])
    for o in range(1, maxorder):
        for gram in pruned_ngrams[o - 1]:
            # Repeat one of the tokens.
            for pos in range(len(gram)):
                if gram[pos] == START_IDX or gram[pos] == END_IDX:
                    continue
                new_gram = gram[:pos] + (gram[pos],) + gram[pos:]

                if new_gram not in all_grams:
                    pruned_ngrams[o].append(new_gram)
                    all_grams.add(new_gram)
    return pruned_ngrams


def parse_lines(lines: List, lexicon: Path) -> List:
    """Parses lines with a lexicon."""
    with open(lexicon, "r") as f:
        lex = (line.strip().split() for line in f)
        lex = {line[0]: line[1:] for line in lex}
        print(len(lex))
    return [[t for w in line.split(WORDSEP) for t in lex[w]] for line in lines]


@click.command()
@click.option("--data_dir", type=str, default=None, help="Path to dataset root.")
@click.option(
    "--tokens", type=str, help="Path to token list (in order used with training)."
)
@click.option("--lexicon", type=str, default=None, help="Path to lexicon")
@click.option(
    "--prune",
    nargs=2,
    type=int,
    help="Threshold values for prune unigrams, bigrams, etc.",
)
@click.option(
    "--blank",
    default=click.Choice(["none", "optional", "forced"]),
    help="Specifies the usage of blank token"
    "'none' - do not use blank token "
    "'optional' - allow an optional blank inbetween tokens"
    "'forced' - force a blank inbetween tokens (also referred to as garbage token)",
)
@click.option("--self_loops", is_flag=True, help="Add self loops for tokens")
@click.option("--disable_backoff", is_flag=True, help="Disable backoff transitions")
@click.option("--save_path", default=None, help="Path to save transition graph.")
def cli(
    data_dir: str,
    tokens: str,
    lexicon: str,
    prune: List[int],
    blank: str,
    self_loops: bool,
    disable_backoff: bool,
    save_path: str,
) -> None:
    """CLI for creating the transitions."""
    logger.info(f"Building {len(prune)}-gram transition models.")

    if data_dir is None:
        data_dir = (
            Path(__file__).resolve().parents[2] / "data" / "processed" / "iam_lines"
        )
        logger.debug(f"Using data dir: {data_dir}")
        if not data_dir.exists():
            raise RuntimeError(f"Could not locate iamdb directory at {data_dir}")
    else:
        data_dir = Path(data_dir)

    # Build table of counts and the back-off if below threshold.
    with open(data_dir / "train.txt", "r") as f:
        lines = [line.strip() for line in f]

    with open(data_dir / tokens, "r") as f:
        tokens = [line.strip() for line in f]

    if lexicon is not None:
        lexicon = data_dir / lexicon
        lines = parse_lines(lines, lexicon)

    tokens_to_idx = {t: e for e, t in enumerate(tokens)}

    ngram = len(prune)

    logger.info("Counting data...")
    ngrams = count_ngrams(lines, ngram, tokens_to_idx)

    pruned_ngrams = prune_ngrams(ngrams, prune)

    for n in range(ngram):
        logger.info(f"Kept {len(pruned_ngrams[n])} of {len(ngrams[n])} {n + 1}-grams")

    if blank == "none":
        pruned_ngrams = add_blank_grams(pruned_ngrams, len(tokens_to_idx), blank)

    if self_loops:
        pruned_ngrams = add_self_loops(pruned_ngrams)

    logger.info("Building graph from pruned ngrams...")
    graph = build_graph(pruned_ngrams, disable_backoff)
    logger.info(f"Graph has {graph.num_arcs()} arcs and {graph.num_nodes()} nodes.")

    save_path = str(data_dir / save_path)

    logger.info(f"Saving graph to {save_path}")
    gtn.save(save_path, graph)


if __name__ == "__main__":
    cli()

"""Transducer and the transducer loss function.py

Stolen from:
    https://github.com/facebookresearch/gtn_applications/blob/master/transducer.py

"""
from pathlib import Path
import itertools
from typing import Dict, List, Optional, Sequence, Set, Tuple

import gtn
import torch
from torch import nn
from torch import Tensor

from text_recognizer.data.utils.iam_preprocessor import Preprocessor


def make_scalar_graph(weight) -> gtn.Graph:
    scalar = gtn.Graph()
    scalar.add_node(True)
    scalar.add_node(False, True)
    scalar.add_arc(0, 1, 0, 0, weight)
    return scalar


def make_chain_graph(sequence) -> gtn.Graph:
    graph = gtn.Graph(False)
    graph.add_node(True)
    for i, s in enumerate(sequence):
        graph.add_node(False, i == (len(sequence) - 1))
        graph.add_arc(i, i + 1, s)
    return graph


def make_transitions_graph(
    ngram: int, num_tokens: int, calc_grad: bool = False
) -> gtn.Graph:
    transitions = gtn.Graph(calc_grad)
    transitions.add_node(True, ngram == 1)

    state_map = {(): 0}

    # First build transitions which include <s>:
    for n in range(1, ngram):
        for state in itertools.product(range(num_tokens), repeat=n):
            in_idx = state_map[state[:-1]]
            out_idx = transitions.add_node(False, ngram == 1)
            state_map[state] = out_idx
            transitions.add_arc(in_idx, out_idx, state[-1])

    for state in itertools.product(range(num_tokens), repeat=ngram):
        state_idx = state_map[state[:-1]]
        new_state_idx = state_map[state[1:]]
        # p(state[-1] | state[:-1])
        transitions.add_arc(state_idx, new_state_idx, state[-1])

    if ngram > 1:
        # Build transitions which include </s>:
        end_idx = transitions.add_node(False, True)
        for in_idx in range(end_idx):
            transitions.add_arc(in_idx, end_idx, gtn.epsilon)

    return transitions


def make_lexicon_graph(
    word_pieces: List, graphemes_to_idx: Dict, special_tokens: Optional[Set]
) -> gtn.Graph:
    """Constructs a graph which transduces letters to word pieces."""
    graph = gtn.Graph(False)
    graph.add_node(True, True)
    for i, wp in enumerate(word_pieces):
        prev = 0
        if special_tokens is not None and wp in special_tokens:
            n = graph.add_node()
            graph.add_arc(prev, n, graphemes_to_idx[wp], i)
        else:
            for character in wp[:-1]:
                n = graph.add_node()
                graph.add_arc(prev, n, graphemes_to_idx[character], gtn.epsilon)
                prev = n
            graph.add_arc(prev, 0, graphemes_to_idx[wp[-1]], i)
    graph.arc_sort()
    return graph


def make_token_graph(
    token_list: List, blank: str = "none", allow_repeats: bool = True
) -> gtn.Graph:
    """Constructs a graph with all the individual token transition models."""
    if not allow_repeats and blank != "optional":
        raise ValueError("Must use blank='optional' if disallowing repeats.")

    ntoks = len(token_list)
    graph = gtn.Graph(False)

    # Creating nodes
    graph.add_node(True, True)
    for i in range(ntoks):
        # We can consume one or more consecutive word
        # pieces for each emission:
        # E.g. [ab, ab, ab] transduces to [ab]
        graph.add_node(False, blank != "forced")

    if blank != "none":
        graph.add_node()

    # Creating arcs
    if blank != "none":
        # Blank index is assumed to be last (ntoks)
        graph.add_arc(0, ntoks + 1, ntoks, gtn.epsilon)
        graph.add_arc(ntoks + 1, 0, gtn.epsilon)

    for i in range(ntoks):
        graph.add_arc((ntoks + 1) if blank == "forced" else 0, i + 1, i)
        graph.add_arc(i + 1, i + 1, i, gtn.epsilon)

        if allow_repeats:
            if blank == "forced":
                # Allow transitions from token to blank only
                graph.add_arc(i + 1, ntoks + 1, ntoks, gtn.epsilon)
            else:
                # Allow transition from token to blank and all other tokens
                graph.add_arc(i + 1, 0, gtn.epsilon)

        else:
            # allow transitions to blank and all other tokens except the same token
            graph.add_arc(i + 1, ntoks + 1, ntoks, gtn.epsilon)
            for j in range(ntoks):
                if i != j:
                    graph.add_arc(i + 1, j + 1, j, j)

    return graph


class TransducerLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        targets,
        tokens,
        lexicon,
        transition_params=None,
        transitions=None,
        reduction="none",
    ) -> Tensor:
        B, T, C = inputs.shape

        losses = [None] * B
        emissions_graphs = [None] * B

        if transitions is not None:
            if transition_params is None:
                raise ValueError("Specified transitions, but not transition params.")

            cpu_data = transition_params.cpu().contiguous()
            transitions.set_weights(cpu_data.data_ptr())
            transitions.calc_grad = transition_params.requires_grad
            transitions.zero_grad()

        def process(b: int) -> None:
            # Create emission graph:
            emissions = gtn.linear_graph(T, C, inputs.requires_grad)
            cpu_data = inputs[b].cpu().contiguous()
            emissions.set_weights(cpu_data.data_ptr())
            target = make_chain_graph(targets[b])
            target.arc_sort(True)

            # Create token tot grapheme decomposition graph
            tokens_target = gtn.remove(gtn.project_output(gtn.compose(target, lexicon)))
            tokens_target.arc_sort()

            # Create alignment graph:
            aligments = gtn.project_input(
                gtn.remove(gtn.compose(tokens, tokens_target))
            )
            aligments.arc_sort()

            # Add transitions scores:
            if transitions is not None:
                aligments = gtn.intersect(transitions, aligments)
                aligments.arc_sort()

            loss = gtn.forward_score(gtn.intersect(emissions, aligments))

            # Normalize if needed:
            if transitions is not None:
                norm = gtn.forward_score(gtn.intersect(emissions, transitions))
                loss = gtn.subtract(loss, norm)

            losses[b] = gtn.negate(loss)

            # Save for backward:
            if emissions.calc_grad:
                emissions_graphs[b] = emissions

        gtn.parallel_for(process, range(B))

        ctx.graphs = (losses, emissions_graphs, transitions)
        ctx.input_shape = inputs.shape

        # Optionally reduce by target length
        if reduction == "mean":
            scales = [(1 / len(t) if len(t) > 0 else 1.0) for t in targets]
        else:
            scales = [1.0] * B

        ctx.scales = scales

        loss = torch.tensor([l.item() * s for l, s in zip(losses, scales)])
        return torch.mean(loss.to(inputs.device))

    @staticmethod
    def backward(ctx, grad_output) -> Tuple:
        losses, emissions_graphs, transitions = ctx.graphs
        scales = ctx.scales

        B, T, C = ctx.input_shape
        calc_emissions = ctx.needs_input_grad[0]
        input_grad = torch.empty((B, T, C)) if calc_emissions else None

        def process(b: int) -> None:
            scale = make_scalar_graph(scales[b])
            gtn.backward(losses[b], scale)
            emissions = emissions_graphs[b]
            if calc_emissions:
                grad = emissions.grad().weights_to_numpy()
                input_grad[b] = torch.tensor(grad).view(1, T, C)

        gtn.parallel_for(process, range(B))

        if calc_emissions:
            input_grad = input_grad.to(grad_output.device)
            input_grad *= grad_output / B

        if ctx.needs_input_grad[4]:
            grad = transitions.grad().weights_to_numpy()
            transition_grad = torch.tensor(grad).to(grad_output.device)
            transition_grad *= grad_output / B
        else:
            transition_grad = None

        return (
            input_grad,
            None,  # target
            None,  # tokens
            None,  # lexicon
            transition_grad,  # transition params
            None,  # transitions graph
            None,
        )


TransducerLoss = TransducerLossFunction.apply


class Transducer(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        ngram: int = 0,
        transitions: str = None,
        blank: str = "none",
        allow_repeats: bool = True,
        reduction: str = "none",
    ) -> None:
        """A generic transducer loss function.

        Args:
            preprocessor (Preprocessor) : The IAM preprocessor for word pieces.
            ngram (int) : Order of the token-level transition model. If `ngram=0`
                then no transition model is used.
            blank (string) : Specifies the usage of blank token
                'none' - do not use blank token
                'optional' - allow an optional blank inbetween tokens
                'forced' - force a blank inbetween tokens (also referred to as garbage token)
            allow_repeats (boolean) : If false, then we don't allow paths with
                consecutive tokens in the alignment graph. This keeps the graph
                unambiguous in the sense that the same input cannot transduce to
                different outputs.
        """
        super().__init__()
        if blank not in ["optional", "forced", "none"]:
            raise ValueError(
                "Invalid value specified for blank. Must be in ['optional', 'forced', 'none']"
            )
        self.tokens = make_token_graph(
            preprocessor.tokens, blank=blank, allow_repeats=allow_repeats
        )
        self.lexicon = make_lexicon_graph(
            preprocessor.tokens,
            preprocessor.graphemes_to_index,
            preprocessor.special_tokens,
        )
        self.ngram = ngram

        self.transitions: Optional[gtn.Graph] = None
        self.transitions_params: Optional[nn.Parameter] = None
        self._load_transitions(transitions, preprocessor, blank)

        if ngram > 0 and transitions is not None:
            raise ValueError("Only one of ngram and transitions may be specified")

        self.reduction = reduction

    def _load_transitions(
        self, transitions: Optional[str], preprocessor: Preprocessor, blank: str
    ):
        """Loads transition graph."""
        processed_path = (
            Path(__file__).resolve().parents[2] / "data" / "processed" / "iam_lines"
        )
        if transitions is not None:
            transitions = gtn.load(str(processed_path / transitions))
        if self.ngram > 0:
            self.transitions = make_transitions_graph(
                self.ngram, len(preprocessor.tokens) + int(blank != "none"), True
            )
        if transitions is not None:
            self.transitions = transitions
            self.transitions.arc_sort()
            self.transitions_params = nn.Parameter(
                torch.zeros(self.transitions.num_arcs())
            )

    def forward(self, inputs: Tensor, targets: Tensor) -> TransducerLoss:
        return TransducerLoss(
            inputs,
            targets,
            self.tokens,
            self.lexicon,
            self.transitions_params,
            self.transitions,
            self.reduction,
        )

    def viterbi(self, outputs: Tensor) -> List[Tensor]:
        B, T, C = outputs.shape

        if self.transitions is not None:
            cpu_data = self.transition_params.cpu().contiguous()
            self.transitions.set_weights(cpu_data.data_ptr())
            self.transitions.calc_grad = False

        self.tokens.arc_sort()

        paths = [None] * B

        def process(b: int) -> None:
            emissions = gtn.linear_graph(T, C, False)
            cpu_data = outputs[b].cpu().contiguous()
            emissions.set_weights(cpu_data.data_ptr())

            if self.transitions is not None:
                full_graph = gtn.intersect(emissions, self.transitions)
            else:
                full_graph = emissions

            # Find the best path and remove back-off arcs:
            path = gtn.remove(gtn.viterbi_path(full_graph))

            # Left compose the viterbi path with the "aligment to token"
            # transducer to get the outputs:
            path = gtn.compose(path, self.tokens)

            # When there are ambiguous paths (allow_repeats is true), we take
            # the shortest:
            path = gtn.viterbi_path(path)
            path = gtn.remove(gtn.project_output(path))
            paths[b] = path.labels_to_list()

        gtn.parallel_for(process, range(B))
        predictions = [torch.IntTensor(path) for path in paths]
        return predictions

"""Greedy decoder."""
from typing import Type

import torch
from torch import Tensor, nn

from text_recognizer.data.tokenizer import Tokenizer


class GreedyDecoder:
    def __init__(
        self,
        network: Type[nn.Module],
        tokenizer: Tokenizer,
        max_output_len: int = 682,
    ) -> None:
        self.network = network
        self.start_index = tokenizer.start_index
        self.end_index = tokenizer.end_index
        self.pad_index = tokenizer.pad_index
        self.max_output_len = max_output_len

    def __call__(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]

        # Encode image(s) to latent vectors.
        img_features = self.network.encode(x)

        # Create a placeholder matrix for storing outputs from the network
        indices = (
            torch.ones((bsz, self.max_output_len), dtype=torch.long, device=x.device)
            * self.pad_index
        )
        indices[:, 0] = self.start_index

        for i in range(1, self.max_output_len):
            tokens = indices[:, :i]  # (B, Sy)
            logits = self.network.decode(tokens, img_features)  # [ B, N, C ]
            indecies_ = logits.argmax(dim=2)  # [ B, N ]
            indices[:, i] = indecies_[:, -1]

            # Early stopping of prediction loop if token is end or padding token.
            if (
                (indices[:, i] == self.end_index) | (indices[:, i] == self.pad_index)
            ).all():
                break

        # Set all tokens after end token to pad token.
        for i in range(1, self.max_output_len):
            idx = (indices[:, i - 1] == self.end_index) | (
                indices[:, i - 1] == self.pad_index
            )
            indices[idx, i] = self.pad_index

        return indices

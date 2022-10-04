"""Greedy decoder."""
from typing import Type
from text_recognizer.data.tokenizer import Tokenizer
import torch
from torch import nn, Tensor


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
        indecies = torch.ones((bsz, self.max_output_len), dtype=torch.long).to(x.device)
        indecies[:, 0] = self.start_index

        for Sy in range(1, self.max_output_len):
            tokens = indecies[:, :Sy]  # (B, Sy)
            logits = self.network.decode(tokens, img_features)  # (B, C, Sy)
            indecies_ = torch.argmax(logits, dim=1)  # (B, Sy)
            indecies[:, Sy : Sy + 1] = indecies_[:, -1:]

            # Early stopping of prediction loop if token is end or padding token.
            if (
                (indecies[:, Sy - 1] == self.end_index)
                | (indecies[:, Sy - 1] == self.pad_index)
            ).all():
                break

        # Set all tokens after end token to pad token.
        for Sy in range(1, self.max_output_len):
            idx = (indecies[:, Sy - 1] == self.end_index) | (
                indecies[:, Sy - 1] == self.pad_index
            )
            indecies[idx, Sy] = self.pad_index

        return indecies

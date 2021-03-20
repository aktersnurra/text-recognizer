"""Implementation of beam search decoder for a sequence to sequence network.

Stolen from: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py

"""
# from typing import List
# from Queue import PriorityQueue

# from loguru import logger
# import torch
# from torch import nn
# from torch import Tensor
# import torch.nn.functional as F


# class Node:
#     def __init__(
#         self, parent: Node, target_index: int, log_prob: Tensor, length: int
#     ) -> None:
#         self.parent = parent
#         self.target_index = target_index
#         self.log_prob = log_prob
#         self.length = length
#         self.reward = 0.0

#     def eval(self, alpha: float = 1.0) -> Tensor:
#         return self.log_prob / (self.length - 1 + 1e-6) + alpha * self.reward


# @torch.no_grad()
# def beam_decoder(
#     network, mapper, device, memory: Tensor = None, max_len: int = 97,
# ) -> Tensor:
#     beam_width = 10
#     topk = 1  # How many sentences to generate.

#     trg_indices = [mapper(mapper.init_token)]

#     end_nodes = []

#     node = Node(None, trg_indices, 0, 1)
#     nodes = PriorityQueue()

#     nodes.put((node.eval(), node))
#     q_size = 1

#     # Beam search
#     for _ in range(max_len):
#         if q_size > 2000:
#             logger.warning("Could not decoder input")
#             break

#         # Fetch the best node.
#         score, n = nodes.get()
#         decoder_input = n.target_index

#         if n.target_index == mapper(mapper.eos_token) and n.parent is not None:
#             end_nodes.append((score, n))

#             # If we reached the maximum number of sentences required.
#             if len(end_nodes) >= 1:
#                 break
#             else:
#                 continue

#         # Forward pass with transformer.
#         trg = torch.tensor(trg_indices, device=device)[None, :].long()
#         trg = network.target_embedding(trg)
#         logits = network.decoder(trg=trg, memory=memory, trg_mask=None)
#         log_prob = F.log_softmax(logits, dim=2)

#         log_prob, indices = torch.topk(log_prob, beam_width)

#         for new_k in range(beam_width):
#             # TODO: continue from here
#             token_index = indices[0][new_k].view(1, -1)
#             log_p = log_prob[0][new_k].item()

#             node = Node()

#             pass

#     pass

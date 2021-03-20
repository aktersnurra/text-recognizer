import torch
from torch import nn

from text_recognizer.networks.transducer import load_transducer_loss, Transducer
import unittest


class TestTransducer(unittest.TestCase):
    def test_viterbi(self):
        T = 5
        N = 4
        B = 2

        # fmt: off
        emissions1 = torch.tensor((
            0, 4, 0, 1,
            0, 2, 1, 1,
            0, 0, 0, 2,
            0, 0, 0, 2,
            8, 0, 0, 2,
            ),
            dtype=torch.float,
        ).view(T, N)
        emissions2 = torch.tensor((
            0, 2, 1, 7,
            0, 2, 9, 1,
            0, 0, 0, 2,
            0, 0, 5, 2,
            1, 0, 0, 2,
            ),
            dtype=torch.float,
        ).view(T, N)
        # fmt: on

        # Test without blank:
        labels = [[1, 3, 0], [3, 2, 3, 2, 3]]
        transducer = Transducer(
            tokens=["a", "b", "c", "d"],
            graphemes_to_idx={"a": 0, "b": 1, "c": 2, "d": 3},
            blank="none",
        )
        emissions = torch.stack([emissions1, emissions2], dim=0)
        predictions = transducer.viterbi(emissions)
        self.assertEqual([p.tolist() for p in predictions], labels)

        # Test with blank without repeats:
        labels = [[1, 0], [2, 2]]
        transducer = Transducer(
            tokens=["a", "b", "c"],
            graphemes_to_idx={"a": 0, "b": 1, "c": 2},
            blank="optional",
            allow_repeats=False,
        )
        emissions = torch.stack([emissions1, emissions2], dim=0)
        predictions = transducer.viterbi(emissions)
        self.assertEqual([p.tolist() for p in predictions], labels)


if __name__ == "__main__":
    unittest.main()

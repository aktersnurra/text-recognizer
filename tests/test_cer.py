"""Test the CER metric."""
import torch

from text_recognizer.models.metrics import CharacterErrorRate


def test_character_error_rate() -> None:
    """Test CER computation."""
    metric = CharacterErrorRate([0, 1])
    preds = torch.Tensor(
        [
            [0, 2, 2, 3, 3, 1],  # error will be 0
            [0, 2, 1, 1, 1, 1],  # error will be 0.75
            [0, 2, 2, 4, 4, 1],  # error will be 0.5
        ]
    )

    targets = torch.Tensor([[0, 2, 2, 3, 3, 1], [0, 2, 2, 3, 3, 1], [0, 2, 2, 3, 3, 1]])
    metric(preds, targets)
    print(metric.compute())
    assert metric.compute() == float(sum([0, 0.75, 0.5]) / 3)



"""CTC loss."""
import torch
from torch import LongTensor, nn, Tensor
import torch.nn.functional as F


class CTCLoss(nn.Module):
    """CTC loss."""

    def __init__(self, blank: int) -> None:
        super().__init__()
        self.blank = blank

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Computes the CTC loss."""
        device = outputs.device

        log_probs = F.log_softmax(outputs, dim=2).permute(1, 0, 2)
        output_lengths = LongTensor([outputs.shape[1]] * outputs.shape[0]).to(device)

        targets_ = LongTensor([]).to(device)
        target_lengths = LongTensor([]).to(device)
        for target in targets:
            # Remove padding
            target = target[target != self.blank].to(device)
            targets_ = torch.cat((targets_, target))
            target_lengths = torch.cat(
                (target_lengths, torch.LongTensor([len(target)]).to(device)), dim=0
            )

        return F.ctc_loss(
            log_probs,
            targets,
            output_lengths,
            target_lengths,
            blank=self.blank,
            zero_infinity=True,
        )

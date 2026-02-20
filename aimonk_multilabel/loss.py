import torch
import torch.nn as nn


class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weights=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=pos_weights
        )

    def forward(self, outputs, targets, mask):
        loss = self.bce(outputs, targets)

        # Ignore NA positions
        loss = loss * mask

        return loss.sum() / mask.sum()

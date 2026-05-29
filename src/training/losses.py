"""Loss functions for binary segmentation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Computes 1 - Dice coefficient on the sigmoid-transformed logits.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        return 1.0 - (
            (2.0 * intersection + self.smooth)
            / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        )


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss.

    This is a common default for medical image segmentation: BCE provides
    a smooth gradient and Dice keeps the loss aligned with the evaluation
    metric.
    """

    def __init__(self, smooth: float = 1.0, dice_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + (1.0 - self.dice_weight) * self.bce(logits, targets)
        )

"""Segmentation metrics: Dice coefficient and IoU."""

import torch


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Compute the Dice coefficient between predicted and target masks.

    Args:
        logits: model output logits of shape (N, 1, H, W)
        targets: ground-truth masks of shape (N, 1, H, W) with values in {0, 1}
        threshold: probability threshold for binarising the prediction
        smooth: smoothing constant to avoid division by zero
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    return (2.0 * intersection + smooth) / (
        preds_flat.sum() + targets_flat.sum() + smooth
    )


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Intersection over Union (Jaccard) score."""
    preds = (torch.sigmoid(logits) > threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

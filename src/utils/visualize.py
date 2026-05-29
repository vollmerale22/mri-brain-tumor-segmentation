"""Utility to visualise model predictions on the validation set."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def save_prediction_grid(
    model,
    loader,
    device,
    output_path: str,
    n_samples: int = 6,
    threshold: float = 0.5,
):
    """
    Save a side-by-side comparison of MRI / ground-truth mask / prediction
    for ``n_samples`` validation images.

    The model is set to eval mode internally.
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    images_collected, masks_collected, preds_collected = [], [], []

    for images, masks in loader:
        images = images.to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits) > threshold).float()

        # De-normalise for display: assume ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        display_images = images * std + mean

        for i in range(display_images.shape[0]):
            if len(images_collected) >= n_samples:
                break
            images_collected.append(display_images[i].cpu().numpy().transpose(1, 2, 0))
            masks_collected.append(masks[i].cpu().numpy())
            preds_collected.append(preds[i, 0].cpu().numpy())

        if len(images_collected) >= n_samples:
            break

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    column_titles = ["MRI input", "Ground truth", "Prediction"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12)

    for row in range(n_samples):
        axes[row, 0].imshow(np.clip(images_collected[row], 0, 1))
        axes[row, 1].imshow(masks_collected[row], cmap="gray")
        axes[row, 2].imshow(preds_collected[row], cmap="gray")
        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction grid to {output_path}")

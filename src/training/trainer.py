"""Training and validation loops."""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import dice_coefficient


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """One epoch of training. Returns the mean training loss."""
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="Train", leave=False)

    for images, masks in progress:
        images = images.to(device)
        masks = masks.float().unsqueeze(1).to(device)  # add channel dim

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one validation pass. Returns (mean loss, mean Dice)."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.float().unsqueeze(1).to(device)
        logits = model(images)

        total_loss += loss_fn(logits, masks).item()
        total_dice += dice_coefficient(logits, masks).item()

    return total_loss / len(loader), total_dice / len(loader)

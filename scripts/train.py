"""
Train a U-Net for brain tumour segmentation on the LGG MRI dataset.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import json
import os
import sys
import time

# Make the ``src`` package importable when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset

from src.data.dataset import LGGDataset
from src.evaluation.metrics import dice_coefficient
from src.models.unet import UNet
from src.training.losses import DiceBCELoss
from src.training.trainer import train_one_epoch, validate
from src.utils.visualize import save_prediction_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def build_transforms(image_size: int):
    train_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=35, p=0.7, border_mode=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build datasets with separate transforms for train and val
    train_tf, val_tf = build_transforms(cfg["image_size"])
    base_for_train = LGGDataset(cfg["data_dir"], transform=train_tf)
    base_for_val = LGGDataset(cfg["data_dir"], transform=val_tf)
    n_total = len(base_for_train)

    # Split indices (same indices for both datasets via shared file ordering)
    indices = np.arange(n_total)
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(indices)
    n_val = int(cfg["val_split"] * n_total)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    train_dataset = Subset(base_for_train, train_idx)
    val_dataset = Subset(base_for_val, val_idx)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # Model, loss, optimizer
    model = UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        features=cfg["features"],
    ).to(device)

    loss_fn = DiceBCELoss(dice_weight=cfg["dice_weight"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Training loop
    os.makedirs(os.path.dirname(cfg["checkpoint_path"]) or ".", exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0
    start = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | {elapsed:.1f}s"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {"model_state_dict": model.state_dict(), "val_dice": val_dice, "epoch": epoch},
                cfg["checkpoint_path"],
            )
            print(f"  -> new best Dice: {val_dice:.4f} (saved)")

    total_time = time.time() - start
    print(f"\nTraining done in {total_time / 60:.1f} min. Best val Dice: {best_dice:.4f}")

    # Save history and final results
    with open(os.path.join(cfg["results_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(cfg["results_dir"], "summary.json"), "w") as f:
        json.dump(
            {
                "best_val_dice": best_dice,
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "epochs": cfg["epochs"],
                "n_train": len(train_dataset),
                "n_val": len(val_dataset),
                "model_params": n_params,
                "device": str(device),
                "training_time_minutes": round(total_time / 60, 2),
            },
            f,
            indent=2,
        )

    # Load best model and create prediction visualisations
    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    save_prediction_grid(
        model, val_loader, device,
        output_path=os.path.join(cfg["results_dir"], "predictions.png"),
        n_samples=6,
    )


if __name__ == "__main__":
    main()

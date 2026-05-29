"""
Evaluate a trained U-Net checkpoint on the validation split.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.dataset import LGGDataset
from src.evaluation.metrics import dice_coefficient, iou_score
from src.models.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    """Compute mean Dice and IoU across the loader."""
    model.eval()
    dice_scores, iou_scores = [], []

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.float().unsqueeze(1).to(device)
        logits = model(images)

        dice_scores.append(dice_coefficient(logits, masks).item())
        iou_scores.append(iou_score(logits, masks).item())

    return {
        "mean_dice": float(np.mean(dice_scores)),
        "std_dice": float(np.std(dice_scores)),
        "mean_iou": float(np.mean(iou_scores)),
        "std_iou": float(np.std(iou_scores)),
        "n_samples": len(dice_scores),
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Reproduce the same val split as in training
    val_tf = A.Compose([
        A.Resize(cfg["image_size"], cfg["image_size"]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    base = LGGDataset(cfg["data_dir"], transform=val_tf)
    indices = np.arange(len(base))
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(indices)
    val_idx = indices[: int(cfg["val_split"] * len(base))].tolist()
    val_dataset = Subset(base, val_idx)
    print(f"Val samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # Load model
    model = UNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        features=cfg["features"],
    ).to(device)
    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
          f"saved val_dice={checkpoint.get('val_dice', '?'):.4f}")

    results = evaluate(model, val_loader, device)
    print("\nValidation results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    os.makedirs(cfg["results_dir"], exist_ok=True)
    out_path = os.path.join(cfg["results_dir"], "evaluation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results to {out_path}")


if __name__ == "__main__":
    main()

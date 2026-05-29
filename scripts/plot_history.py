"""
Plot training history (loss and Dice curves) from results/history.json.

Usage:
    python scripts/plot_history.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt


def main():
    history_path = "results/history.json"
    if not os.path.exists(history_path):
        sys.exit(f"No history found at {history_path}. Did training finish?")

    with open(history_path) as f:
        history = json.load(f)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss", marker="o", markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Val loss", marker="s", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and validation loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_dice"], color="tab:green", marker="o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice coefficient")
    ax2.set_title("Validation Dice score")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "results/training_curves.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


if __name__ == "__main__":
    main()

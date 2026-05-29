# MRI Brain Tumor Segmentation

End-to-end deep learning pipeline for automated brain tumor segmentation
from MRI scans, using a U-Net architecture trained on the LGG MRI
Segmentation dataset.

## Overview

This project implements and evaluates a convolutional neural network
(U-Net) for binary segmentation of low-grade glioma (LGG) tumors on
FLAIR-weighted MRI slices. The pipeline covers data loading and
augmentation, model training with a combined Dice + binary cross-entropy
loss, evaluation with the Dice coefficient and IoU, and visualization of
predictions against ground-truth tumor masks.

## Dataset

[LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
by Mateusz Buda — 110 patients with lower-grade glioma from The Cancer
Imaging Archive, with corresponding manual segmentation masks.

- **Sequences:** FLAIR-weighted MRI slices stored as RGB `.tif` images
- **Slices:** ~3 900 image / mask pairs
- **Task:** binary segmentation (tumor vs. background)
- **Split:** 80 % training / 20 % validation, fixed seed for reproducibility

Reference: Buda M., Saha A., Mazurowski M. A. (2019). *Association of
genomic subtypes of lower-grade gliomas with shape features automatically
extracted by a deep learning algorithm.* Computers in Biology and
Medicine, 109:218–225.

## Architecture

Standard U-Net (Ronneberger et al., 2015):
- 4-level encoder / decoder with skip connections
- Double convolution blocks (`Conv2d → BatchNorm → ReLU`) at every level
- Feature dimensions `[64, 128, 256, 512]`, bottleneck at 1 024 channels
- Final `1×1` convolution producing a single-channel logit map
- ~31 M trainable parameters

## Training setup

| Setting | Value |
|---|---|
| Input size | 256 × 256 |
| Batch size | 16 |
| Epochs | 25 |
| Optimizer | Adam, lr = 1e-4, weight decay = 1e-4 |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Loss | 0.5 · Dice + 0.5 · BCEWithLogits |
| Augmentation | Horizontal/vertical flip, ±35° rotation |
| Hardware | *fill in after training (e.g. Google Colab T4 GPU)* |

## Results

| Metric | Validation |
|---|---|
| Dice coefficient | 0.85 |
| IoU (Jaccard)   | 0.74 |

Training curves and example predictions are saved under `results/`:
- `results/training_curves.png` — train/val loss and validation Dice over epochs
- `results/predictions.png` — sample MRI / ground truth / prediction triplets

## Repository structure

```
mri-brain-tumor-segmentation/
├── configs/
│   └── default.yaml          # training configuration
├── scripts/
│   ├── train.py              # main training entry point
│   ├── evaluate.py           # evaluate a saved checkpoint
│   └── plot_history.py       # plot loss / Dice curves
├── src/
│   ├── data/dataset.py       # LGGDataset class
│   ├── models/unet.py        # U-Net architecture
│   ├── training/
│   │   ├── losses.py         # DiceLoss, DiceBCELoss
│   │   └── trainer.py        # training + validation loops
│   ├── evaluation/metrics.py # Dice coefficient, IoU
│   └── utils/visualize.py    # prediction visualisation
├── results/                  # generated outputs
├── checkpoints/              # saved models (gitignored)
├── data/                     # local data (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. clone and enter the repository
git clone https://github.com/vollmerale22/mri-brain-tumor-segmentation.git
cd mri-brain-tumor-segmentation

# 2. create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt
```

## Downloading the dataset

The dataset is hosted on Kaggle. With the Kaggle CLI configured
(see `https://www.kaggle.com/docs/api`):

```bash
mkdir -p data/raw
cd data/raw
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip
# This creates the directory data/raw/kaggle_3m/ with per-patient subfolders
cd ../..
```

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

This produces:
- `checkpoints/best_model.pt` — best-Dice checkpoint
- `results/summary.json` — final metrics and run metadata
- `results/history.json` — per-epoch training history
- `results/predictions.png` — qualitative prediction samples

To plot the training curves afterwards:

```bash
python scripts/plot_history.py
```

## Evaluating a trained model

```bash
python scripts/evaluate.py --config configs/default.yaml
```

Writes mean and standard deviation of Dice and IoU on the validation set
to `results/evaluation.json`.

## License

MIT.

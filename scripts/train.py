"""
Main training script
Demonstrates: End-to-end ML pipeline, configuration management
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import torch.optim as optim
import yaml
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.unet import UNet3D
from src.data.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.training.losses import CombinedLoss


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 60)
    print("MRI Brain Tumor Segmentation - Training")
    print("=" * 60)
    
    # Set device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    dataloaders = create_dataloaders(
        data_dir=config['data']['processed_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    train_loader = dataloaders.get('train')
    val_loader = dataloaders.get('val')
    
    if train_loader is None or val_loader is None:
        print("Warning: No data found. This is expected if you haven't downloaded the dataset yet.")
        print("The training pipeline is ready to use once you add data!")
        return
    
    # Create model
    print("\nInitializing model...")
    model = UNet3D( in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features']
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create loss function
    criterion = CombinedLoss()
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=config['data']['num_classes'],
        checkpoint_dir=config['paths']['checkpoints'],
        log_dir=config['paths']['logs'],
        max_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MRI segmentation model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args)
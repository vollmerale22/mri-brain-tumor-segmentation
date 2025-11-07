"""
Training loop for medical image segmentation
Demonstrates: PyTorch training, model optimization, logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm
import numpy as np

from src.training.losses import CombinedLoss
from src.training.metrics import compute_batch_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for medical image segmentation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_classes: int,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        max_epochs: int = 100,
        early_stopping_patience: int = 15
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.patience_counter = 0
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_metrics = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs} [Train]")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Compute metrics (on a subset to save time)
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    metrics = compute_batch_metrics(outputs, masks, self.num_classes)
                    all_metrics.append(metrics)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = self._average_metrics(all_metrics)
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        all_metrics = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs} [Val]")
        
        with torch.no_grad():
            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Track metrics
                total_loss += loss.item()
                metrics = compute_batch_metrics(outputs, masks, self.num_classes)
                all_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Average metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = self._average_metrics(all_metrics)
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def _average_metrics(self, metrics_list):
        """Average a list of metric dictionaries."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if not np.isinf(m[key])]
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Checkpoint not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        logger.info(f"Checkpoint loaded: {filepath}")
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Check for improvement
            val_loss = val_metrics['loss']
            val_dice = val_metrics.get('mean_dice', 0.0)
            
            improved = False
            
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                improved = True
                self.save_checkpoint('best_model_dice.pth')
                logger.info(f"✓ New best Dice score: {val_dice:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True
                self.save_checkpoint('best_model_loss.pth')
                logger.info(f"✓ New best validation loss: {val_loss:.4f}")
            
            # Early stopping
            if improved:
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation Dice: {self.best_val_dice:.4f}")
        
        self.writer.close()
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to TensorBoard."""
        epoch = self.current_epoch
        
        # Log losses
        self.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        # Log Dice scores
        if 'mean_dice' in train_metrics and 'mean_dice' in val_metrics:
            self.writer.add_scalars('Dice', {
                'train': train_metrics['mean_dice'],
                'val': val_metrics['mean_dice']
            }, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)


if __name__ == "__main__":
    print("Trainer class ready!")
    print("This module is meant to be imported and used in training scripts.")
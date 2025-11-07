"""
Loss functions for medical image segmentation
Demonstrates: PyTorch, mathematical understanding, domain knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Dice coefficient is commonly used in medical imaging to measure
    overlap between predicted and ground truth segmentation.
    
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W, D) - logits from model
            targets: (B, H, W, D) - ground truth labels
        """
        # Convert logits to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()
        
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), predictions.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets_one_hot).sum(dim=2)
        union = predictions.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - Dice as loss (we want to minimize)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combination of Cross Entropy and Dice Loss.
    
    This is commonly used in medical imaging as it combines:
    - CE: good for pixel-wise classification
    - Dice: good for handling class imbalance
    """
    def __init__(self, weight_ce: float = 0.5, weight_dice: float = 0.5):
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Focuses training on hard examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size, num_classes, depth, height, width = 2, 4, 32, 32, 32
    predictions = torch.randn(batch_size, num_classes, depth, height, width)
    targets = torch.randint(0, num_classes, (batch_size, depth, height, width))
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    loss = dice_loss(predictions, targets)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss()
    loss = combined_loss(predictions, targets)
    print(f"Combined Loss: {loss.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    loss = focal_loss(predictions, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    print("✓ All loss functions working!")
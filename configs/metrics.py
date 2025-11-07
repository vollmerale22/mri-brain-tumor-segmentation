"""
Evaluation metrics for medical image segmentation
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from typing import Dict, List


class SegmentationMetrics:
    """
    Compute various metrics for segmentation evaluation.
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def dice_coefficient(self, pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
        """
        Compute Dice Similarity Coefficient for a specific class.
        
        DSC = 2 * |X ∩ Y| / (|X| + |Y|)
        """
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        if union == 0:
            return 1.0  # Both empty, perfect match
        
        return 2.0 * intersection / union
    
    def iou(self, pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
        """
        Compute Intersection over Union (Jaccard Index).
        
        IoU = |X ∩ Y| / |X ∪ Y|
        """
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    def sensitivity(self, pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
        """
        Compute Sensitivity (Recall, True Positive Rate).
        
        Sensitivity = TP / (TP + FN)
        """
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        tp = np.logical_and(pred_mask, target_mask).sum()
        fn = np.logical_and(~pred_mask, target_mask).sum()
        
        if (tp + fn) == 0:
            return 1.0
        
        return tp / (tp + fn)
    
    def specificity(self, pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
        """
        Compute Specificity (True Negative Rate).
        
        Specificity = TN / (TN + FP)
        """
        pred_mask = (pred == class_id)
        target_mask = (target == class_id)
        
        tn = np.logical_and(~pred_mask, ~target_mask).sum()
        fp = np.logical_and(pred_mask, ~target_mask).sum()
        
        if (tn + fp) == 0:
            return 1.0
        
        return tn / (tn + fp)
    
    def hausdorff_distance(self, pred: np.ndarray, target: np.ndarray, class_id: int, percentile: int = 95) -> float:
        """
        Compute Hausdorff Distance (HD95).
        
        Measures maximum distance between boundary points.
        HD95 is the 95th percentile, more robust to outliers.
        """
        pred_mask = (pred == class_id).astype(np.uint8)
        target_mask = (target == class_id).astype(np.uint8)
        
        if pred_mask.sum() == 0 or target_mask.sum() == 0:
            return np.inf
        
        # Compute distance transforms
        pred_dist = distance_transform_edt(~pred_mask)
        target_dist = distance_transform_edt(~target_mask)
        
        # Get surface points
        pred_surface = pred_mask - np.logical_and(pred_mask, distance_transform_edt(pred_mask) > 1)
        target_surface = target_mask - np.logical_and(target_mask, distance_transform_edt(target_mask) > 1)
        
        # Distances from pred surface to target
        # Distances from pred surface to target
        pred_distances = pred_dist[target_surface > 0]
        target_distances = target_dist[pred_surface > 0]
        
        if len(pred_distances) == 0 or len(target_distances) == 0:
            return np.inf
        
        # Compute percentile
        hd_pred = np.percentile(pred_distances, percentile)
        hd_target = np.percentile(target_distances, percentile)
        
        return max(hd_pred, hd_target)
    
    def compute_all_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Compute all metrics for all classes.
        
        Returns:
            Dictionary with metrics for each class
        """
        metrics = {}
        
        for class_id in range(self.num_classes):
            class_name = f"class_{class_id}"
            
            metrics[f"{class_name}_dice"] = self.dice_coefficient(pred, target, class_id)
            metrics[f"{class_name}_iou"] = self.iou(pred, target, class_id)
            metrics[f"{class_name}_sensitivity"] = self.sensitivity(pred, target, class_id)
            metrics[f"{class_name}_specificity"] = self.specificity(pred, target, class_id)
            metrics[f"{class_name}_hd95"] = self.hausdorff_distance(pred, target, class_id)
        
        # Compute mean metrics (excluding background class 0)
        metrics["mean_dice"] = np.mean([metrics[f"class_{i}_dice"] for i in range(1, self.num_classes)])
        metrics["mean_iou"] = np.mean([metrics[f"class_{i}_iou"] for i in range(1, self.num_classes)])
        metrics["mean_sensitivity"] = np.mean([metrics[f"class_{i}_sensitivity"] for i in range(1, self.num_classes)])
        metrics["mean_specificity"] = np.mean([metrics[f"class_{i}_specificity"] for i in range(1, self.num_classes)])
        
        return metrics


def compute_batch_metrics(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    Compute metrics for a batch of predictions.
    
    Args:
        predictions: (B, C, H, W, D) - logits
        targets: (B, H, W, D) - ground truth
        num_classes: Number of classes
    
    Returns:
        Dictionary of averaged metrics
    """
    # Convert predictions to class labels
    pred_labels = torch.argmax(predictions, dim=1)
    
    # Move to CPU and convert to numpy
    pred_labels = pred_labels.cpu().numpy()
    targets = targets.cpu().numpy()
    
    metrics_calculator = SegmentationMetrics(num_classes)
    
    batch_metrics = []
    for i in range(pred_labels.shape[0]):
        metrics = metrics_calculator.compute_all_metrics(pred_labels[i], targets[i])
        batch_metrics.append(metrics)
    
    # Average across batch
    avg_metrics = {}
    for key in batch_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in batch_metrics])
    
    return avg_metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing segmentation metrics...")
    
    # Create dummy predictions and targets
    pred = np.random.randint(0, 4, size=(64, 64, 64))
    target = np.random.randint(0, 4, size=(64, 64, 64))
    
    metrics_calc = SegmentationMetrics(num_classes=4)
    metrics = metrics_calc.compute_all_metrics(pred, target)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float) and not np.isinf(value):
            print(f"{key}: {value:.4f}")
    
    print("✓ Metrics computation working!")
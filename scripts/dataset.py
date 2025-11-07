"""
MRI Brain Tumor Dataset Loader
Demonstrates: Medical imaging, PyTorch, data handling, NumPy
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BraTSDataset(Dataset):
    """
    Brain Tumor Segmentation Dataset
    
    Handles multi-modal MRI scans (T1, T1ce, T2, FLAIR) with segmentation masks.
    
    Args:
        data_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        modalities: List of MRI modalities to load
        transform: Optional data augmentation
        normalize: Whether to normalize intensities
    """
    
    MODALITIES = ['t1', 't1ce', 't2', 'flair']
    LABELS = {
        0: 'background',
        1: 'necrotic_core',
        2: 'edema',
        3: 'enhancing_tumor'
    }
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        modalities: List[str] = None,
        transform=None,
        normalize: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.modalities = modalities or self.MODALITIES
        self.transform = transform
        self.normalize = normalize
        
        # Validate modalities
        for mod in self.modalities:
            if mod not in self.MODALITIES:
                raise ValueError(f"Invalid modality: {mod}")
        
        # Load file paths
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self)} {split} samples with modalities: {self.modalities}")
    
    def _load_samples(self) -> pd.DataFrame:
        """
        Load dataset samples from directory structure.
        
        Expected structure:
        data_dir/
            patient_001/
                t1.nii.gz
                t1ce.nii.gz
                t2.nii.gz
                flair.nii.gz
                seg.nii.gz
        """
        samples = []
        patient_dirs = sorted([d for d in self.data_dir.glob('*/') if d.is_dir()])
        
        for patient_dir in patient_dirs:
            sample = {'patient_id': patient_dir.name}
            
            # Check all modalities exist
            valid = True
            for modality in self.modalities:
                file_path = patient_dir / f"{modality}.nii.gz"
                if not file_path.exists():
                    logger.warning(f"Missing {modality} for {patient_dir.name}")
                    valid = False
                    break
                sample[modality] = str(file_path)
            
            # Check segmentation exists
            seg_path = patient_dir / "seg.nii.gz"
            if not seg_path.exists():
                logger.warning(f"Missing segmentation for {patient_dir.name}")
                valid = False
            else:
                sample['segmentation'] = str(seg_path)
            
            if valid:
                samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample (image, mask) pair.
        
        Returns:
            image: Tensor of shape (C, H, W, D) where C = number of modalities
            mask: Tensor of shape (H, W, D) with integer labels
        """
        sample = self.samples.iloc[idx]
        
        # Load all modalities
        images = []
        for modality in self.modalities:
            img_path = sample[modality]
            img = self._load_nifti(img_path)
            
            if self.normalize:
                img = self._normalize(img)
            
            images.append(img)
        
        # Stack modalities along channel dimension
        image = np.stack(images, axis=0)  # Shape: (C, H, W, D)
        
        # Load segmentation mask
        mask = self._load_nifti(sample['segmentation'])
        
        # Apply transformations if any
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Convert to PyTorch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def _load_nifti(self, file_path: str) -> np.ndarray:
        """Load NIfTI file and return numpy array."""
        nifti = nib.load(file_path)
        data = nifti.get_fdata()
        return data.astype(np.float32)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensities to zero mean and unit variance.
        Only normalize non-zero regions (brain tissue).
        """
        mask = image > 0
        if mask.sum() > 0:
            mean = image[mask].mean()
            std = image[mask].std()
            if std > 0:
                image[mask] = (image[mask] - mean) / std
        return image
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Compute dataset statistics.
        Demonstrates: Pandas, NumPy, statistical analysis
        """
        stats = []
        
        for idx in range(len(self)):
            sample = self.samples.iloc[idx]
            
            # Load mask to get tumor statistics
            mask = self._load_nifti(sample['segmentation'])
            
            # Compute volumes for each class
            for label, name in self.LABELS.items():
                volume = (mask == label).sum()
                stats.append({
                    'patient_id': sample['patient_id'],
                    'class': name,
                    'label': label,
                    'volume_voxels': volume
                })
        
        df = pd.DataFrame(stats)
        return df


def create_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    splits: List[str] = ['train', 'val', 'test']
) -> dict:
    """
    Create DataLoaders for all splits.
    
    Args:
        data_dir: Path to dataset
        batch_size: Batch size for training
        num_workers: Number of parallel workers
        splits: Which splits to create loaders for
    
    Returns:
        Dictionary of DataLoaders
    """
    dataloaders = {}
    
    for split in splits:
        dataset = BraTSDataset(data_dir=data_dir, split=split)
        
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        dataloaders[split] = dataloader
        
        logger.info(f"Created {split} DataLoader: {len(dataset)} samples, "
                   f"{len(dataloader)} batches")
    
    return dataloaders


if __name__ == "__main__":
    # Test the dataset
    print("Testing BraTSDataset...")
    
    # This would fail without actual data, but shows usage
    try:
        dataset = BraTSDataset(data_dir="data/processed", split="train")
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            image, mask = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Unique labels in mask: {torch.unique(mask)}")
            
            # Get statistics
            stats = dataset.get_statistics()
            print("\nDataset Statistics:")
            print(stats.groupby('class')['volume_voxels'].describe())
    
    except Exception as e:
        print(f"Note: Dataset test failed (expected without real data): {e}")
        print("Dataset class is ready to use once data is available!")
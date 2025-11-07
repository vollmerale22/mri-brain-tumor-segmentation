"""
Test the dataset loader
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import BraTSDataset

def test_dataset():
    print("="*60)
    print("Testing Dataset Loader")
    print("="*60)
    
    # Test loading
    dataset = BraTSDataset(
        data_dir='data/processed',
        split='train',
        normalize=True
    )
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Number of samples: {len(dataset)}")
    
    if len(dataset) > 0:
        # Load first sample
        image, mask = dataset[0]
        
        print(f"\n✓ Sample loaded successfully!")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Mask dtype: {mask.dtype}")
        print(f"  Unique labels in mask: {mask.unique().tolist()}")
        
        # Get statistics
        print("\nComputing dataset statistics...")
        stats = dataset.get_statistics()
        print("\nTumor volume statistics:")
        print(stats.groupby('class')['volume_voxels'].describe())
    else:
        print("\n❌ No samples found!")
        print("Make sure you've run:")
        print("  1. python scripts/generate_synthetic_data.py")
        print("  2. python scripts/organize_data.py")

if __name__ == "__main__":
    test_dataset()
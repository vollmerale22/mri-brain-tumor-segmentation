"""
Organize data into train/val/test splits
"""

import shutil
from pathlib import Path
import random
import json

def split_data(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into train/validation/test sets.
    
    Args:
        source_dir: Directory containing patient folders
        output_dir: Output directory for organized data
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
    """
    
    source = Path(source_dir)
    output = Path(output_dir)
    
    # Get all patient directories
    patient_dirs = sorted([d for d in source.glob('patient_*') if d.is_dir()])
    
    if len(patient_dirs) == 0:
        print(f"❌ No patient directories found in {source}")
        print("Make sure you've generated synthetic data first!")
        return
    
    print("="*60)
    print("Organizing Data into Train/Val/Test Splits")
    print("="*60)
    print(f"\nFound {len(patient_dirs)} patients")
    
    # Shuffle
    random.seed(42)
    random.shuffle(patient_dirs)
    
    # Calculate split indices
    n_total = len(patient_dirs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_patients = patient_dirs[:n_train]
    val_patients = patient_dirs[n_train:n_train + n_val]
    test_patients = patient_dirs[n_train + n_val:]
    
    print(f"\nSplit:")
    print(f"  Training:   {len(train_patients)} patients ({train_ratio*100:.0f}%)")
    print(f"  Validation: {len(val_patients)} patients ({val_ratio*100:.0f}%)")
    print(f"  Test:       {len(test_patients)} patients ({test_ratio*100:.0f}%)")
    
    # Create directories and copy files
    splits = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }
    
    for split_name, patients in splits.items():
        split_dir = output / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {split_name} data...")
        for patient_dir in patients:
            # Copy entire patient directory
            dest = split_dir / patient_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(patient_dir, dest)
    
    # Save split information
    split_info = {
        'train': [p.name for p in train_patients],
        'val': [p.name for p in val_patients],
        'test': [p.name for p in test_patients]
    }
    
    info_file = output / 'split_info.json'
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Data organization complete!")
    print(f"✓ Output directory: {output}")
    print(f"✓ Split info saved: {info_file}")
    
    # Verify structure
    print("\nVerifying structure...")
    for split in ['train', 'val', 'test']:
        split_dir = output / split
        n_patients = len(list(split_dir.glob('patient_*')))
        print(f"  {split}: {n_patients} patients")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize data into splits')
    parser.add_argument('--source', type=str, default='data/raw/synthetic',
                        help='Source directory with patient data')
    parser.add_argument('--output', type=str, default='data/processed',
                        help='Output directory for organized data')
    parser.add_argument('--train', type=float, default=0.7,
                        help='Training split ratio')
    parser.add_argument('--val', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test', type=float, default=0.15,
                        help='Test split ratio')
    
    args = parser.parse_args()
    
    split_data(args.source, args.output, args.train, args.val, args.test)
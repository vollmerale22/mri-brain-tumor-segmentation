"""
Download sample MRI data for testing
"""

import os
import urllib.request
import tarfile
from pathlib import Path
import shutil

def download_file(url, destination):
    """Download file with progress bar"""
    print(f"Downloading from {url}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        print(f"\rProgress: {percent:.1f}%", end='')
    
    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    print("\n✓ Download complete!")

def download_sample_data():
    """Download Medical Decathlon sample data"""
    
    # Create directories
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downloading Sample MRI Dataset")
    print("="*60)
    
    # Medical Decathlon - Task01 Brain Tumour
    url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
    tar_path = data_dir / "Task01_BrainTumour.tar"
    
    if not tar_path.exists():
        download_file(url, tar_path)
    else:
        print(f"✓ File already exists: {tar_path}")
    
    # Extract
    print("\nExtracting archive...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(data_dir)
    
    print("✓ Extraction complete!")
    
    # Show what we downloaded
    extracted_dir = data_dir / "Task01_BrainTumour"
    if extracted_dir.exists():
        train_dir = extracted_dir / "imagesTr"
        label_dir = extracted_dir / "labelsTr"
        
        num_images = len(list(train_dir.glob("*.nii.gz"))) if train_dir.exists() else 0
        num_labels = len(list(label_dir.glob("*.nii.gz"))) if label_dir.exists() else 0
        
        print(f"\n✓ Downloaded {num_images} MRI scans")
        print(f"✓ Downloaded {num_labels} segmentation masks")
        print(f"\nLocation: {extracted_dir}")
    
    # Clean up tar file to save space
    print("\nCleaning up...")
    tar_path.unlink()
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)

if __name__ == "__main__":
    download_sample_data()
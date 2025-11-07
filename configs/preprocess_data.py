"""
Preprocess downloaded data into expected format
"""

from pathlib import Path
import shutil
from tqdm import tqdm

def organize_brats_data(source_dir, target_dir):
    """
    Reorganize BraTS data into our expected structure.
    """
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    
    patient_dirs = sorted(source.glob("BraTS*"))
    
    print(f"Found {len(patient_dirs)} patients")
    
    for patient_dir in tqdm(patient_dirs):
        # Create target directory
        patient_name = patient_dir.name
        new_dir = target / patient_name
        new_dir.mkdir(exist_ok=True)
        
        # Copy and rename files
        files = {
            't1': patient_dir / f"{patient_name}_t1.nii.gz",
            't1ce': patient_dir / f"{patient_name}_t1ce.nii.gz",
            't2': patient_dir / f"{patient_name}_t2.nii.gz",
            'flair': patient_dir / f"{patient_name}_flair.nii.gz",
            'seg': patient_dir / f"{patient_name}_seg.nii.gz"
        }
        
        for modality, source_file in files.items():
            if source_file.exists():
                target_file = new_dir / f"{modality}.nii.gz"
                shutil.copy(source_file, target_file)
    
    print(f"✓ Organized data in {target}")

if __name__ == "__main__":
    # Adjust paths as needed
    organize_brats_data("data/raw/MICCAI_BraTS2020_TrainingData", 
                       "data/processed")
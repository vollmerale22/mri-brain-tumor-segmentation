"""
Generate synthetic MRI data for testing the pipeline
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def generate_synthetic_mri(output_dir, num_samples=10):
    """
    Generate synthetic MRI data that mimics real structure.
    
    Creates T1, T1ce, T2, FLAIR modalities and segmentation masks.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating Synthetic MRI Data")
    print("="*60)
    
    # Image dimensions
    depth, height, width = 128, 128, 128
    
    for i in tqdm(range(num_samples), desc="Creating samples"):
        patient_dir = output_dir / f"patient_{i:03d}"
        patient_dir.mkdir(exist_ok=True)
        
        # Create base brain structure
        brain_mask = create_brain_mask(depth, height, width)
        
        # Generate modalities with different contrasts
        t1 = generate_modality(brain_mask, contrast='t1')
        t1ce = generate_modality(brain_mask, contrast='t1ce')
        t2 = generate_modality(brain_mask, contrast='t2')
        flair = generate_modality(brain_mask, contrast='flair')
        
        # Generate tumor segmentation
        seg = generate_tumor_segmentation(depth, height, width)
        
        # Save as NIfTI files
        save_nifti(t1, patient_dir / "t1.nii.gz")
        save_nifti(t1ce, patient_dir / "t1ce.nii.gz")
        save_nifti(t2, patient_dir / "t2.nii.gz")
        save_nifti(flair, patient_dir / "flair.nii.gz")
        save_nifti(seg, patient_dir / "seg.nii.gz")
    
    print(f"\n✓ Generated {num_samples} synthetic MRI samples")
    print(f"Location: {output_dir}")
    print("\nYou can now test your pipeline with this data!")

def create_brain_mask(depth, height, width):
    """Create a realistic brain-shaped mask"""
    z, y, x = np.ogrid[:depth, :height, :width]
    center_z, center_y, center_x = depth//2, height//2, width//2
    
    # Ellipsoid shape for brain
    mask = ((z - center_z)**2 / (depth*0.35)**2 + 
            (y - center_y)**2 / (height*0.4)**2 + 
            (x - center_x)**2 / (width*0.35)**2) <= 1
    
    return mask.astype(float)

def generate_modality(brain_mask, contrast='t1'):
    """Generate MRI modality with specific contrast"""
    
    # Base intensity
    if contrast == 't1':
        base_intensity = 100
    elif contrast == 't1ce':
        base_intensity = 120
    elif contrast == 't2':
        base_intensity = 150
    else:  # flair
        base_intensity = 130
    
    # Create image
    image = np.random.randn(*brain_mask.shape) * 10 + base_intensity
    image = image * brain_mask  # Apply brain mask
    image = np.maximum(image, 0)  # No negative values
    
    # Add some texture
    from scipy.ndimage import gaussian_filter
    image = gaussian_filter(image, sigma=2)
    
    return image.astype(np.float32)

def generate_tumor_segmentation(depth, height, width):
    """Generate random tumor segmentation"""
    seg = np.zeros((depth, height, width), dtype=np.uint8)
    
    # Random tumor location
    center_z = np.random.randint(depth//4, 3*depth//4)
    center_y = np.random.randint(height//4, 3*height//4)
    center_x = np.random.randint(width//4, 3*width//4)
    
    z, y, x = np.ogrid[:depth, :height, :width]
    
    # Enhancing tumor (class 3)
    radius_enh = np.random.randint(8, 15)
    mask_enh = ((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2) <= radius_enh**2
    seg[mask_enh] = 3
    
    # Edema (class 2)
    radius_edema = radius_enh + np.random.randint(5, 10)
    mask_edema = ((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2) <= radius_edema**2
    seg[mask_edema & ~mask_enh] = 2
    
    # Necrotic core (class 1)
    radius_nec = radius_enh // 2
    mask_nec = ((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2) <= radius_nec**2
    seg[mask_nec] = 1
    
    return seg

def save_nifti(data, filepath):
    """Save numpy array as NIfTI file"""
    nifti_img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nifti_img, filepath)

if __name__ == "__main__":
    # Generate 20 synthetic samples
    generate_synthetic_mri("data/raw/synthetic", num_samples=20)
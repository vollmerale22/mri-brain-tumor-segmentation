"""
Generate synthetic MRI data for testing the pipeline
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

def generate_synthetic_mri(output_dir, num_samples=20):
    """
    Generate synthetic MRI data that mimics real structure.
    
    Creates T1, T1ce, T2, FLAIR modalities and segmentation masks.
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating Synthetic MRI Data")
    print("="*60)
    
    # Image dimensions (smaller for faster generation)
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
    print(f"✓ Location: {output_dir}")
    print(f"✓ Total size: ~{num_samples * 20}MB")
    print("\nYou can now use this data to test your pipeline!")


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
    
    # Base intensity varies by modality
    intensity_map = {
        't1': 100,
        't1ce': 120,
        't2': 150,
        'flair': 130
    }
    
    base_intensity = intensity_map.get(contrast, 100)
    
    # Create image with noise
    image = np.random.randn(*brain_mask.shape) * 15 + base_intensity
    image = image * brain_mask  # Apply brain mask
    image = np.maximum(image, 0)  # No negative values
    
    # Add smooth texture (requires scipy)
    try:
        from scipy.ndimage import gaussian_filter
        image = gaussian_filter(image, sigma=1.5)
    except ImportError:
        pass  # Skip smoothing if scipy not available
    
    return image.astype(np.float32)


def generate_tumor_segmentation(depth, height, width):
    """Generate random tumor segmentation with realistic structure"""
    seg = np.zeros((depth, height, width), dtype=np.uint8)
    
    # Random tumor location (avoid edges)
    center_z = np.random.randint(depth//3, 2*depth//3)
    center_y = np.random.randint(height//3, 2*height//3)
    center_x = np.random.randint(width//3, 2*width//3)
    
    z, y, x = np.ogrid[:depth, :height, :width]
    
    # Create distance from center
    dist = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
    
    # Enhancing tumor (class 3) - innermost
    radius_enh = np.random.randint(5, 12)
    seg[dist <= radius_enh] = 3
    
    # Necrotic core (class 1) - very center
    radius_nec = radius_enh * 0.4
    seg[dist <= radius_nec] = 1
    
    # Edema (class 2) - surrounding
    radius_edema = radius_enh + np.random.randint(8, 15)
    seg[(dist > radius_enh) & (dist <= radius_edema)] = 2
    
    return seg


def save_nifti(data, filepath):
    """Save numpy array as NIfTI file"""
    # Create simple affine matrix (identity with 1mm spacing)
    affine = np.eye(4)
    
    nifti_img = nib.Nifti1Image(data, affine=affine)
    nib.save(nifti_img, filepath)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic MRI data')
    parser.add_argument('--output', type=str, default='data/raw/synthetic',
                        help='Output directory for synthetic data')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate')
    
    args = parser.parse_args()
    
    generate_synthetic_mri(args.output, args.num_samples)
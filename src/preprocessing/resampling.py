
from pathlib import Path
import SimpleITK as sitk

# Dataset location: outside the GitHub repository
dataset_dir = (
    Path.home()
    / "Downloads"
    / "Task07_Pancreas"
    / "Task07_Pancreas"
)

ct_path = dataset_dir / "imagesTr" / "pancreas_421.nii.gz"
mask_path = dataset_dir / "labelsTr" / "pancreas_421.nii.gz"

# Save processed images outside the GitHub repository
output_dir = dataset_dir / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# Check dataset files
if not ct_path.exists():
    raise FileNotFoundError(f"CT image not found: {ct_path}")

if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

# Load CT image and segmentation mask
ct = sitk.ReadImage(str(ct_path))
mask = sitk.ReadImage(str(mask_path))

print("Original CT size:", ct.GetSize())
print("Original CT spacing:", ct.GetSpacing())
print("Original mask size:", mask.GetSize())
print("Original mask spacing:", mask.GetSpacing())

# Target voxel spacing in millimeters (x, y, z)
target_spacing = (1.0, 1.0, 2.0)

# Calculate the resampled CT dimensions
original_size = ct.GetSize()
original_spacing = ct.GetSpacing()

new_size = [
    int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
    for i in range(3)
]

print("Target spacing:", target_spacing)
print("New CT size:", new_size)

# Resample CT using linear interpolation
ct_resampler = sitk.ResampleImageFilter()
ct_resampler.SetOutputSpacing(target_spacing)
ct_resampler.SetSize(new_size)
ct_resampler.SetOutputDirection(ct.GetDirection())
ct_resampler.SetOutputOrigin(ct.GetOrigin())
ct_resampler.SetTransform(sitk.Transform())
ct_resampler.SetDefaultPixelValue(-1024)
ct_resampler.SetInterpolator(sitk.sitkLinear)

resampled_ct = ct_resampler.Execute(ct)

# Resample mask to exactly match the resampled CT geometry
mask_resampler = sitk.ResampleImageFilter()
mask_resampler.SetReferenceImage(resampled_ct)
mask_resampler.SetTransform(sitk.Transform())
mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
mask_resampler.SetDefaultPixelValue(0)

resampled_mask = mask_resampler.Execute(mask)

# Save processed images outside the GitHub repository
ct_output = output_dir / "pancreas_421_ct_resampled.nii.gz"
mask_output = output_dir / "pancreas_421_mask_resampled.nii.gz"

sitk.WriteImage(resampled_ct, str(ct_output))
sitk.WriteImage(resampled_mask, str(mask_output))

# Verify the results
mask_array = sitk.GetArrayViewFromImage(resampled_mask)

print("\nResampling completed successfully!")
print("Resampled CT size:", resampled_ct.GetSize())
print("Resampled CT spacing:", resampled_ct.GetSpacing())
print("Resampled mask size:", resampled_mask.GetSize())
print("Resampled mask spacing:", resampled_mask.GetSpacing())
print("Mask labels:", sorted(set(mask_array.flatten().tolist())))
print("CT saved to:", ct_output)
print("Mask saved to:", mask_output)

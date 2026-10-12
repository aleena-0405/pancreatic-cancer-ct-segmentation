
import torch
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
)

data = {
    "image": (
        r"C:\Users\ALEENA ANTONY\Downloads\Task07_Pancreas"
        r"\Task07_Pancreas\imagesTr\pancreas_421.nii.gz"
    ),
    "label": (
        r"C:\Users\ALEENA ANTONY\Downloads\Task07_Pancreas"
        r"\Task07_Pancreas\labelsTr\pancreas_421.nii.gz"
    ),
}

# 1. Load the CT and its matching mask
data = LoadImaged(
    keys=["image", "label"],
    image_only=True,
)(data)

# 2. Add a channel dimension
data = EnsureChannelFirstd(
    keys=["image", "label"],
    channel_dim="no_channel",
)(data)

# 3. Standardize orientation
data = Orientationd(
    keys=["image", "label"],
    axcodes="RAS",
)(data)

# 4. Resample to the candidate spacing
data = Spacingd(
    keys=["image", "label"],
    pixdim=(1.0, 1.0, 2.0),
    mode=("bilinear", "nearest"),
)(data)

# 5. Normalize CT intensity values
data = ScaleIntensityRanged(
    keys=["image"],
    a_min=-100,
    a_max=400,
    b_min=0.0,
    b_max=1.0,
    clip=True,
)(data)

# 6. Check the results
ct = data["image"]
mask = data["label"]

print("Normalized CT shape:", tuple(ct.shape))
print("Normalized CT minimum:", ct.min().item())
print("Normalized CT maximum:", ct.max().item())
print("Mask labels:", torch.unique(mask).tolist())

print("Normalization test completed.")

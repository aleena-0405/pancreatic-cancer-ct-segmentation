
import os
import numpy as np
import torch

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
)

# --------------------------------------------------
# 1. Dataset paths
# --------------------------------------------------

# Dataset path; override with MSD_TASK07_DIR when needed
DATASET_DIR = os.environ.get(
    "MSD_TASK07_DIR",
    os.path.join(
        os.path.expanduser("~"),
        "Downloads",
        "Task07_Pancreas",
        "Task07_Pancreas",
    ),
)
IMAGE_DIR = os.path.join(DATASET_DIR, "imagesTr")
LABEL_DIR = os.path.join(DATASET_DIR, "labelsTr")

# Test three additional cases
CASE_IDS = ["pancreas_004", "pancreas_005", "pancreas_006"]

# --------------------------------------------------
# 2. Define the preprocessing pipeline
# --------------------------------------------------

preprocessing = Compose([
    LoadImaged(
        keys=["image", "label"],
        image_only=True,
    ),
    EnsureChannelFirstd(
        keys=["image", "label"],
        channel_dim="no_channel",
    ),
    Orientationd(
        keys=["image", "label"],
        axcodes="RAS",
    ),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 2.0),
        mode=("bilinear", "nearest"),
    ),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-100,
        a_max=400,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
])

# --------------------------------------------------
# 3. Validate each case
# --------------------------------------------------

results = []

for case_id in CASE_IDS:
    print("\n" + "=" * 50)
    print("Testing:", case_id)
    print("=" * 50)

    image_path = os.path.join(
        IMAGE_DIR, case_id + ".nii.gz"
    )
    label_path = os.path.join(
        LABEL_DIR, case_id + ".nii.gz"
    )

    # Check that both files exist
    if not os.path.isfile(image_path):
        print("FAILED: CT image not found.")
        results.append((case_id, False))
        continue

    if not os.path.isfile(label_path):
        print("FAILED: Matching mask not found.")
        results.append((case_id, False))
        continue

    try:
        data = {
            "image": image_path,
            "label": label_path,
        }

        # Apply the same transformations to both files
        data = preprocessing(data)

        ct = data["image"]
        mask = data["label"]

        # Print basic information
        print("CT shape:", tuple(ct.shape))
        print("Mask shape:", tuple(mask.shape))
        print("CT minimum:", ct.min().item())
        print("CT maximum:", ct.max().item())
        print("Mask labels:", torch.unique(mask).tolist())

        # Perform validation checks
        shape_match = ct.shape == mask.shape

        intensity_valid = (
            torch.isfinite(ct).all().item()
            and ct.min().item() >= 0.0
            and ct.max().item() <= 1.0
        )

        mask_values = torch.unique(mask).tolist()
        mask_valid = (
            len(mask_values) > 0
            and all(
                value in (0.0, 1.0, 2.0)
                for value in mask_values
            )
        )

        mass_present = (mask == 2).any().item()

        print("\nValidation checks:")
        print("CT/mask shapes match:", shape_match)
        print("CT intensity range valid:", intensity_valid)
        print("Mask labels valid:", mask_valid)
        print("Pancreatic mass present:", mass_present)

        passed = (
            shape_match
            and intensity_valid
            and mask_valid
            and mass_present
        )

        if passed:
            print("\nRESULT: PASSED")
        else:
            print("\nRESULT: CHECK REQUIRED")

        results.append((case_id, passed))

    except Exception as error:
        print("ERROR:", error)
        results.append((case_id, False))

# --------------------------------------------------
# 4. Print the final summary
# --------------------------------------------------

print("\n" + "=" * 50)
print("FINAL PREPROCESSING VALIDATION SUMMARY")
print("=" * 50)

for case_id, passed in results:
    status = "PASSED" if passed else "CHECK REQUIRED"
    print(f"{case_id}: {status}")

passed_count = sum(
    passed for _, passed in results
)

print(
    f"\nCases passed: {passed_count}/{len(results)}"
)
print("Validation run completed.")

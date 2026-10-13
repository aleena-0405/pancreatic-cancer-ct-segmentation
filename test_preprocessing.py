
import os
import numpy as np
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
)

# --------------------------------------------------
# 1. Dataset and output paths
# --------------------------------------------------

# Override this environment variable if your dataset is
# stored somewhere other than the default Downloads folder.
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

# Generated PNGs stay local and are excluded by .gitignore.
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "preprocessing_results",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# These are the three additional cases in this test run.
CASE_IDS = [
    "pancreas_004",
    "pancreas_005",
    "pancreas_006",
]

# Provisional candidate settings: validate before full-dataset use.
CANDIDATE_SPACING = (1.0, 1.0, 2.0)
CANDIDATE_INTENSITY_WINDOW = (-100, 400)

# --------------------------------------------------
# 2. Define the reusable preprocessing pipeline
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
        labels=None,
    ),
    Spacingd(
        keys=["image", "label"],
        pixdim=CANDIDATE_SPACING,
        mode=("bilinear", "nearest"),
    ),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=CANDIDATE_INTENSITY_WINDOW[0],
        a_max=CANDIDATE_INTENSITY_WINDOW[1],
        b_min=0.0,
        b_max=1.0,
        clip=True,
    ),
])

# --------------------------------------------------
# 3. Validate and visualize each case
# --------------------------------------------------

results = []

for case_id in CASE_IDS:
    print("\n" + "=" * 55)
    print(f"Processing: {case_id}")
    print("=" * 55)

    image_path = os.path.join(
        IMAGE_DIR, f"{case_id}.nii.gz"
    )
    label_path = os.path.join(
        LABEL_DIR, f"{case_id}.nii.gz"
    )

    if not os.path.isfile(image_path):
        print(f"ERROR: CT image not found: {image_path}")
        results.append((case_id, False))
        continue

    if not os.path.isfile(label_path):
        print(f"ERROR: Matching mask not found: {label_path}")
        results.append((case_id, False))
        continue

    try:
        data = preprocessing({
            "image": image_path,
            "label": label_path,
        })

        # Convert channel-first MONAI tensors to NumPy arrays.
        ct = data["image"][0].detach().cpu().numpy()
        mask = data["label"][0].detach().cpu().numpy()

        # ------------------------------------------
        # 4. Automated validation checks
        # ------------------------------------------

        shape_match = ct.shape == mask.shape

        intensity_valid = (
            np.isfinite(ct).all()
            and ct.min() >= 0.0
            and ct.max() <= 1.0
        )

        mask_labels = np.unique(mask)
        mask_valid = np.isin(
            mask_labels, [0, 1, 2]
        ).all()

        mass_slices = np.where(
            np.any(mask == 2, axis=(0, 1))
        )[0]
        mass_present = len(mass_slices) > 0

        print("CT shape:", ct.shape)
        print("Mask shape:", mask.shape)
        print("CT minimum:", float(ct.min()))
        print("CT maximum:", float(ct.max()))
        print("Mask labels:", mask_labels.tolist())
        print("CT/mask shapes match:", shape_match)
        print("Intensity range valid:", intensity_valid)
        print("Mask labels valid:", mask_valid)
        print("Mass label 2 present:", mass_present)

        # Do not create an overlay if basic checks fail.
        if not (
            shape_match
            and intensity_valid
            and mask_valid
            and mass_present
        ):
            print("RESULT: CHECK REQUIRED")
            results.append((case_id, False))
            continue

        # ------------------------------------------
        # 5. Select a representative mass-containing slice
        # ------------------------------------------

        slice_index = int(
            mass_slices[len(mass_slices) // 2]
        )

        ct_slice = ct[:, :, slice_index]
        mask_slice = mask[:, :, slice_index]

        # ------------------------------------------
        # 6. Save the CT-only PNG
        # ------------------------------------------

        ct_path = os.path.join(
            OUTPUT_DIR, f"{case_id}_ct.png"
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(
            ct_slice,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"{case_id}: CT only - slice {slice_index}"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            ct_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # ------------------------------------------
        # 7. Save the segmentation overlay PNG
        # ------------------------------------------

        overlay_path = os.path.join(
            OUTPUT_DIR, f"{case_id}_overlay.png"
        )

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.imshow(
            ct_slice,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        # Label 1: pancreas. Label 2: pancreatic mass.
        pancreas_overlay = np.ma.masked_where(
            mask_slice != 1,
            mask_slice,
        )
        mass_overlay = np.ma.masked_where(
            mask_slice != 2,
            mask_slice,
        )

        ax.imshow(
            pancreas_overlay,
            cmap="Greens",
            alpha=0.45,
            vmin=0,
            vmax=2,
        )
        ax.imshow(
            mass_overlay,
            cmap="autumn",
            alpha=0.8,
            vmin=0,
            vmax=2,
        )

        ax.set_title(
            f"{case_id}: Pancreas and mass\n"
            f"Slice {slice_index}"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            overlay_path,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print("Saved CT:", ct_path)
        print("Saved overlay:", overlay_path)
        print("RESULT: AUTOMATED CHECKS PASSED")

        results.append((case_id, True))

    except Exception as error:
        print(f"ERROR processing {case_id}: {error}")
        results.append((case_id, False))

# --------------------------------------------------
# 8. Print final summary
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 55)

for case_id, passed in results:
    status = (
        "AUTOMATED CHECKS PASSED"
        if passed
        else "CHECK REQUIRED"
    )
    print(f"{case_id}: {status}")

passed_count = sum(
    passed for _, passed in results
)

print(f"\nCases passed: {passed_count}/{len(results)}")
print("Candidate spacing:", CANDIDATE_SPACING)
print("Candidate intensity window:", CANDIDATE_INTENSITY_WINDOW)
print("Output folder:", OUTPUT_DIR)
print("Preprocessing test completed.")

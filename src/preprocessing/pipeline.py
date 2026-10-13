
from pathlib import Path

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
)


# --------------------------------------------------
# 1. Preprocessing configuration
# --------------------------------------------------

TARGET_SPACING = (1.0, 1.0, 2.0)

CT_MIN = -100.0
CT_MAX = 400.0

NORMALIZED_MIN = 0.0
NORMALIZED_MAX = 1.0


# --------------------------------------------------
# 2. Create the preprocessing pipeline
# --------------------------------------------------

def get_preprocessing_transform():
    """
    Create the preprocessing transforms for a CT image
    and its corresponding segmentation mask.

    The CT and mask are transformed together to preserve
    their spatial correspondence.
    """

    return Compose([
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
            pixdim=TARGET_SPACING,
            mode=("bilinear", "nearest"),
        ),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=CT_MIN,
            a_max=CT_MAX,
            b_min=NORMALIZED_MIN,
            b_max=NORMALIZED_MAX,
            clip=True,
        ),
    ])


# --------------------------------------------------
# 3. Validate one case
# --------------------------------------------------

def preprocess_case(image_path, label_path):
    """
    Preprocess one CT image and its matching mask.

    Parameters
    ----------
    image_path : str or Path
        Path to the CT image (.nii or .nii.gz).

    label_path : str or Path
        Path to the corresponding segmentation mask.

    Returns
    -------
    dict
        Preprocessed image and label as MONAI tensors.
    """

    image_path = Path(image_path)
    label_path = Path(label_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"CT image not found: {image_path}"
        )

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Segmentation mask not found: {label_path}"
        )

    transform = get_preprocessing_transform()

    data = transform({
        "image": str(image_path),
        "label": str(label_path),
    })

    image = data["image"]
    label = data["label"]

    # Verify matching tensor shapes
    if image.shape != label.shape:
        raise ValueError(
            "CT and mask shapes do not match after preprocessing: "
            f"{tuple(image.shape)} versus {tuple(label.shape)}"
        )

    # Verify image values are finite and normalized
    image_array = image.detach().cpu().numpy()

    if not np.isfinite(image_array).all():
        raise ValueError(
            "Preprocessed CT contains non-finite values."
        )

    if image_array.min() < NORMALIZED_MIN or (
        image_array.max() > NORMALIZED_MAX
    ):
        raise ValueError(
            "Preprocessed CT values are outside [0, 1]."
        )

    # Check that nearest-neighbor interpolation preserved
    # the expected segmentation label values.
    label_array = label.detach().cpu().numpy()
    label_values = np.unique(label_array)

    if not np.isin(label_values, [0, 1, 2]).all():
        raise ValueError(
            f"Unexpected segmentation labels: {label_values.tolist()}"
        )

    return data


# --------------------------------------------------
# 4. Run a small test when this file is executed
# --------------------------------------------------


if __name__ == "__main__":

    dataset_dir = (
        Path.home()
        / "Downloads"
        / "Task07_Pancreas"
        / "Task07_Pancreas"
    )

    case_ids = [
        "pancreas_421",
        "pancreas_001",
        "pancreas_004",
        "pancreas_005",
        "pancreas_006",
    ]

    results = []

    for case_id in case_ids:
        print("\n" + "=" * 45)
        print(f"Testing: {case_id}")
        print("=" * 45)

        image_path = (
            dataset_dir / "imagesTr" / f"{case_id}.nii.gz"
        )
        label_path = (
            dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
        )

        try:
            data = preprocess_case(image_path, label_path)

            image = data["image"]
            label = data["label"]

            print("CT shape:", tuple(image.shape))
            print("Mask shape:", tuple(label.shape))
            print("CT minimum:", image.min().item())
            print("CT maximum:", image.max().item())

            labels = np.unique(
                label.detach().cpu().numpy()
            ).tolist()

            print("Mask labels:", labels)

            mass_present = (label == 2).any().item()
            print("Pancreatic mass present:", mass_present)

            if not mass_present:
                print("WARNING: No mass label found.")

            results.append((case_id, True))

        except Exception as error:
            print("FAILED:", error)
            results.append((case_id, False))

    print("\n" + "=" * 45)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 45)

    for case_id, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{case_id}: {status}")

    print(
        f"\nCases passed: "
        f"{sum(passed for _, passed in results)}/{len(results)}"
    )

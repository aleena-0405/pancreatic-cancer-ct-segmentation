from pathlib import Path

import torch
from monai.transforms import RandCropByPosNegLabeld

from src.preprocessing.pipeline import preprocess_case


def main():
    # Dataset location
    dataset_dir = (
        Path.home()
        / "Downloads"
        / "Task07_Pancreas"
        / "Task07_Pancreas"
    )

    case_id = "pancreas_421"

    image_path = dataset_dir / "imagesTr" / f"{case_id}.nii.gz"
    label_path = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"

    print("Loading and preprocessing:", case_id)

    data = preprocess_case(image_path, label_path)

    # Keep only the pancreatic mass as the target.
    data["label"] = (data["label"] == 2).float()

    print("Full CT shape:", tuple(data["image"].shape))
    print("Full mask shape:", tuple(data["label"].shape))
    print("Mass voxels:", int(data["label"].sum().item()))

    # Extract one patch containing mass when possible.
    crop_transform = RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 32),
        pos=1,
        neg=1,
        num_samples=1,
        image_key="image",
        image_threshold=0,
    )

    patches = crop_transform(data)
    patch = patches[0]

    image_patch = patch["image"]
    mask_patch = patch["label"]

    print("\nPatch test results")
    print("CT patch shape:", tuple(image_patch.shape))
    print("Mask patch shape:", tuple(mask_patch.shape))
    print("CT value range:",
          float(image_patch.min()),
          "to",
          float(image_patch.max()))
    print("Mask labels:", torch.unique(mask_patch).tolist())
    print("Mass voxels in patch:", int(mask_patch.sum().item()))

    assert image_patch.shape == (1, 96, 96, 32)
    assert mask_patch.shape == (1, 96, 96, 32)
    assert torch.isfinite(image_patch).all()
    assert set(torch.unique(mask_patch).tolist()).issubset({0.0, 1.0})

    print("\nPASS: The CT and binary mass-mask patches are valid.")


if __name__ == "__main__":
    main()
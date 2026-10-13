from pathlib import Path

import torch
from monai.losses import DiceCELoss

from src.preprocessing.pipeline import preprocess_case
from src.segmentation.model import create_model


def main():
    # Use GPU when available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset_dir = (
        Path.home()
        / "Downloads"
        / "Task07_Pancreas"
        / "Task07_Pancreas"
    )

    case_id = "pancreas_421"

    image_path = dataset_dir / "imagesTr" / f"{case_id}.nii.gz"
    label_path = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"

    # Load and preprocess one labelled CT case.
    data = preprocess_case(image_path, label_path)

    # Binary target: mass = 1, everything else = 0.
    image = data["image"]
    label = (data["label"] == 2).float()

    # Choose a real mass voxel as the patch center.
    mass_coords = torch.nonzero(label[0] == 1)

    if len(mass_coords) == 0:
        raise ValueError(f"No mass found in {case_id}")

    center = mass_coords[len(mass_coords) // 2].tolist()
    x, y, z = center

    spatial_shape = image.shape[1:]
    patch_size = (96, 96, 32)

    starts = [
        max(0, min(center_value - size // 2, dim - size))
        for center_value, size, dim
        in zip(center, patch_size, spatial_shape)
    ]

    image_patch = image[
        :,
        starts[0]:starts[0] + 96,
        starts[1]:starts[1] + 96,
        starts[2]:starts[2] + 32,
    ]

    mask_patch = label[
        :,
        starts[0]:starts[0] + 96,
        starts[1]:starts[1] + 96,
        starts[2]:starts[2] + 32,
    ]

    # Add the batch dimension and move data to the GPU.
    image_batch = image_patch.unsqueeze(0).to(device)
    mask_batch = mask_patch.unsqueeze(0).to(device)

    if mask_batch.sum().item() == 0:
        raise ValueError("Selected patch contains no mass voxels.")

    # Create the model and optimizer.
    model = create_model().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = DiceCELoss(sigmoid=True)

    # One complete training step.
    optimizer.zero_grad()

    logits = model(image_batch)
    loss = loss_function(logits, mask_batch)

    if not torch.isfinite(loss):
        raise ValueError("Loss is not finite.")

    loss.backward()
    optimizer.step()

    print("Device:", device)
    print("Input shape:", tuple(image_batch.shape))
    print("Target shape:", tuple(mask_batch.shape))
    print("Mass voxels in patch:", int(mask_batch.sum().item()))
    print("Initial training-step loss:", round(loss.item(), 4))
    print("PASS: One optimizer update completed.")


if __name__ == "__main__":
    main()
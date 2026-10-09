
import os
from pathlib import Path

import torch
from monai.transforms import LoadImage


def main():
    # Set MSD_TASK07_DIR to your local Task07_Pancreas directory.
    dataset_dir = Path(
        os.environ.get(
            "MSD_TASK07_DIR",
            str(
                Path.home()
                / "Downloads"
                / "Task07_Pancreas"
                / "Task07_Pancreas"
            ),
        )
    )

    ct_path = dataset_dir / "imagesTr" / "pancreas_421.nii.gz"
    mask_path = dataset_dir / "labelsTr" / "pancreas_421.nii.gz"

    # Check that both files exist before loading.
    for path in (ct_path, mask_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    loader = LoadImage(image_only=True)

    ct = loader(str(ct_path))
    mask = loader(str(mask_path))

    print("CT shape:", tuple(ct.shape))
    print("Mask shape:", tuple(mask.shape))
    print("Mask labels:", torch.unique(torch.as_tensor(mask)).tolist())
    print("CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()


import os
from pathlib import Path

# Dataset root can be overridden using MSD_TASK07_DIR.
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

images_path = dataset_dir / "imagesTr"

if not images_path.is_dir():
    raise FileNotFoundError(f"Training images directory not found: {images_path}")

files = sorted(
    path.name
    for path in images_path.glob("*.nii.gz")
    if not path.name.startswith("._")
)

print("Dataset directory:", images_path)
print("Number of CT cases:", len(files))
print("First 5 cases:", files[:5])
print("Last 5 cases:", files[-5:])

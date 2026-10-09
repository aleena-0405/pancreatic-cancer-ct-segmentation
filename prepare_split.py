
import os
import random
from pathlib import Path

# Dataset root can be overridden using the MSD_TASK07_DIR environment variable.
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
labels_path = dataset_dir / "labelsTr"

if not images_path.is_dir():
    raise FileNotFoundError(f"Training images directory not found: {images_path}")

if not labels_path.is_dir():
    raise FileNotFoundError(f"Training labels directory not found: {labels_path}")

# Collect CT files and match each with its segmentation mask.
ct_files = sorted(
    f.name
    for f in images_path.glob("*.nii.gz")
    if not f.name.startswith("._")
)

valid_cases = []

for filename in ct_files:
    mask_path = labels_path / filename

    if mask_path.is_file():
        valid_cases.append(filename)
    else:
        print("Missing mask:", filename)

# Reproducible 80/20 split.
random.seed(42)
random.shuffle(valid_cases)

split_index = int(len(valid_cases) * 0.8)

train_cases = valid_cases[:split_index]
val_cases = valid_cases[split_index:]

# Verify there is no overlap between the two splits.
assert set(train_cases).isdisjoint(val_cases)

print("Total valid cases:", len(valid_cases))
print("Training cases:", len(train_cases))
print("Validation cases:", len(val_cases))
print("Overlapping cases:", len(set(train_cases) & set(val_cases)))

# Save only case identifiers, not the medical images.
splits_dir = Path("splits")
splits_dir.mkdir(parents=True, exist_ok=True)

(splits_dir / "train.txt").write_text(
    "\n".join(train_cases) + "\n",
    encoding="utf-8",
)
(splits_dir / "val.txt").write_text(
    "\n".join(val_cases) + "\n",
    encoding="utf-8",
)

print("Split files saved in:", splits_dir.resolve())

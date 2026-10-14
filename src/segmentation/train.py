
from pathlib import Path
import csv
import random

import numpy as np
import torch
from monai.losses import DiceCELoss

from src.preprocessing.pipeline import preprocess_case
from src.segmentation.model import create_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
PATCH_SIZE = (96, 96, 32)
EPOCHS = 1
LEARNING_RATE = 1e-4
CHECKPOINT_EVERY = 10

DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "Task07_Pancreas"
    / "Task07_Pancreas"
)

PROJECT_DIR = Path.cwd()
OUTPUT_DIR = Path.home() / "pancreas_training_checkpoints"

RECOVERY_PATH = OUTPUT_DIR / "baseline_recovery.pt"
BEST_CHECKPOINT_PATH = OUTPUT_DIR / "baseline_best.pt"
METRICS_PATH = OUTPUT_DIR / "baseline_metrics.csv"

CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "val_patch_loss",
    "val_patch_dice",
]


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. DATASET AND SPLIT HELPERS
# ============================================================

def get_cases(split_name):
    split_file = PROJECT_DIR / "splits" / f"{split_name}.txt"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    cases = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]

    if not cases:
        raise ValueError(f"No cases found in {split_file}")

    return cases


def validate_dataset_paths(train_cases, val_cases):
    if set(train_cases) & set(val_cases):
        raise ValueError(
            "Training and validation splits overlap."
        )

    for case_id in train_cases + val_cases:
        image_path = DATASET_DIR / "imagesTr" / case_id
        label_path = DATASET_DIR / "labelsTr" / case_id

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        if not label_path.exists():
            raise FileNotFoundError(label_path)


def load_case(case_id):
    """Preprocess one CT image and its matching segmentation mask."""
    return preprocess_case(
        DATASET_DIR / "imagesTr" / case_id,
        DATASET_DIR / "labelsTr" / case_id,
    )


# ============================================================
# 4. PATCH EXTRACTION
# ============================================================

def extract_patch(image, mask, require_mass):
    """
    Extract a 3D patch.

    Positive patch: contains at least one mass voxel.
    Negative patch: contains no mass voxels.

    Mask label 2 represents the target mass.
    """

    spatial_shape = image.shape[1:]

    if any(
        dim < size
        for dim, size in zip(spatial_shape, PATCH_SIZE)
    ):
        raise ValueError(
            f"Volume shape {spatial_shape} is smaller "
            f"than patch size {PATCH_SIZE}."
        )

    if require_mass:
        coords = torch.nonzero(mask[0] > 0)
    else:
        coords = torch.nonzero(mask[0] == 0)

    if coords.numel() == 0:
        raise ValueError(
            f"No suitable center found. require_mass={require_mass}"
        )

    attempts = min(50, len(coords))

    for _ in range(attempts):
        center = coords[
            random.randrange(len(coords))
        ].tolist()

        starts = [
            max(
                0,
                min(c - size // 2, dim - size),
            )
            for c, size, dim in zip(
                center, PATCH_SIZE, spatial_shape
            )
        ]

        slices = tuple(
            slice(start, start + size)
            for start, size in zip(starts, PATCH_SIZE)
        )

        image_patch = image[(slice(None),) + slices]
        mask_patch = mask[(slice(None),) + slices]

        mass_count = int(mask_patch.sum().item())

        if require_mass and mass_count > 0:
            return image_patch, mask_patch

        if not require_mass and mass_count == 0:
            return image_patch, mask_patch

    raise RuntimeError(
        "Unable to extract a suitable patch after repeated attempts."
    )


def get_patch(case_id, require_mass):
    data = load_case(case_id)

    image = data["image"]
    mask = (data["label"] == 2).float()

    return extract_patch(image, mask, require_mass)


# ============================================================
# 5. METRICS AND SAVING
# ============================================================

def calculate_dice(logits, target):
    prediction = torch.sigmoid(logits) >= 0.5
    target_binary = target >= 0.5

    intersection = (
        prediction & target_binary
    ).sum().item()

    denominator = (
        prediction.sum().item()
        + target_binary.sum().item()
    )

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def save_checkpoint(path, model, optimizer, epoch,
                    processed_cases, train_cases,
                    val_cases, best_val_dice,
                    train_losses=None):
    """Save enough state to resume training."""

    checkpoint = {
        "epoch": epoch,
        "processed_cases": processed_cases,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "seed": SEED,
        "train_cases": train_cases,
        "validation_cases": val_cases,
        "best_val_dice": best_val_dice,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "train_losses": train_losses or [],
    }

    temporary_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def append_metrics(epoch, train_loss, val_loss, val_dice):
    """Append completed epoch results without deleting earlier results."""

    file_exists = METRICS_PATH.exists()

    with METRICS_PATH.open("a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
        )

        if not file_exists or METRICS_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_patch_loss": val_loss,
            "val_patch_dice": val_dice,
        })


# ============================================================
# 6. TRAINING
# ============================================================

def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_cases = get_cases("train")
    val_cases = get_cases("val")

    validate_dataset_paths(train_cases, val_cases)

    print("=" * 60)
    print("PANCREATIC MASS SEGMENTATION - BASELINE TRAINING")
    print("=" * 60)
    print("Device:", device)
    print("Training cases:", len(train_cases))
    print("Validation cases:", len(val_cases))
    print("Epochs:", EPOCHS)
    print("Patch size:", PATCH_SIZE)
    print("Checkpoint interval:", CHECKPOINT_EVERY)
    print("Output directory:", OUTPUT_DIR)

    model = create_model().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = DiceCELoss(sigmoid=True)

    start_epoch = 1
    start_case_index = 0
    best_val_dice = -1.0
    train_losses = []
    shuffled_cases = None

    # --------------------------------------------------------
    # Resume from recovery checkpoint if requested.
    # Set RESUME_TRAINING = True below to enable this.
    # --------------------------------------------------------
    RESUME_TRAINING = False

    if RESUME_TRAINING and RECOVERY_PATH.exists():
        checkpoint = torch.load(
            RECOVERY_PATH,
            map_location=device,
            weights_only=False,
        )

        if checkpoint["train_cases"] != train_cases:
            raise ValueError(
                "Training case list differs from checkpoint."
            )

        if checkpoint["validation_cases"] != val_cases:
            raise ValueError(
                "Validation case list differs from checkpoint."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"]
        start_case_index = checkpoint["processed_cases"]
        best_val_dice = checkpoint["best_val_dice"]
        train_losses = checkpoint.get("train_losses", [])
        shuffled_cases = checkpoint["train_cases"]

        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])

        if (
            torch.cuda.is_available()
            and checkpoint.get("cuda_rng_state") is not None
        ):
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_state"]
            )

        print(
            f"Resuming epoch {start_epoch} after "
            f"{start_case_index} training cases."
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()

        if shuffled_cases is None or epoch != start_epoch:
            shuffled_cases = train_cases.copy()
            random.shuffle(shuffled_cases)
            start_case_index = 0
            train_losses = []

        print(f"\nStarting epoch {epoch}/{EPOCHS}")

        for index in range(start_case_index, len(shuffled_cases)):
            case_id = shuffled_cases[index]

            try:
                # Preprocess each case once during this epoch.
                data = load_case(case_id)
                image = data["image"]
                mask = (data["label"] == 2).float()

                for require_mass in (True, False):
                    image_patch, mask_patch = extract_patch(
                        image,
                        mask,
                        require_mass,
                    )

                    images = image_patch.unsqueeze(0).to(device)
                    targets = mask_patch.unsqueeze(0).to(device)

                    optimizer.zero_grad(set_to_none=True)

                    logits = model(images)
                    loss = loss_function(logits, targets)

                    if not torch.isfinite(loss).item():
                        raise ValueError(
                            f"Non-finite loss for {case_id}"
                        )

                    loss.backward()
                    optimizer.step()

                    train_losses.append(float(loss.detach().cpu()))

                completed = index + 1

                if (
                    completed % CHECKPOINT_EVERY == 0
                    or completed == len(shuffled_cases)
                ):
                    print(
                        f"Epoch {epoch}: processed "
                        f"{completed}/{len(shuffled_cases)} cases",
                        flush=True,
                    )

                    save_checkpoint(
                        RECOVERY_PATH,
                        model,
                        optimizer,
                        epoch,
                        completed,
                        shuffled_cases,
                        val_cases,
                        best_val_dice,
                        train_losses,
                    )

            except Exception:
                print(
                    f"\nTraining failed while processing {case_id}. "
                    "Check the error below."
                )
                raise

        mean_train_loss = float(np.mean(train_losses))

        # ----------------------------------------------------
        # VALIDATION
        # One mass-centered patch per validation case.
        # This is not whole-volume evaluation.
        # ----------------------------------------------------
        model.eval()
        val_losses = []
        val_dices = []

        print("\nStarting validation...")

        with torch.no_grad():
            for index, case_id in enumerate(val_cases, start=1):
                image_patch, mask_patch = get_patch(
                    case_id,
                    require_mass=True,
                )

                images = image_patch.unsqueeze(0).to(device)
                targets = mask_patch.unsqueeze(0).to(device)

                logits = model(images)
                loss = loss_function(logits, targets)

                val_losses.append(float(loss.cpu()))
                val_dices.append(
                    calculate_dice(logits, targets)
                )

                if index % 10 == 0 or index == len(val_cases):
                    print(
                        f"Validation: {index}/{len(val_cases)} cases",
                        flush=True,
                    )

        mean_val_loss = float(np.mean(val_losses))
        mean_val_dice = float(np.mean(val_dices))

        print(f"\nEpoch {epoch} results")
        print("Mean training loss:", round(mean_train_loss, 4))
        print("Mean validation patch loss:", round(mean_val_loss, 4))
        print("Mean validation patch Dice:", round(mean_val_dice, 4))

        append_metrics(
            epoch,
            mean_train_loss,
            mean_val_loss,
            mean_val_dice,
        )

        if mean_val_dice > best_val_dice:
            best_val_dice = mean_val_dice

            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                model,
                optimizer,
                epoch,
                len(shuffled_cases),
                shuffled_cases,
                val_cases,
                best_val_dice,
                train_losses,
            )

            print("Saved best model:", BEST_CHECKPOINT_PATH)

        # Mark this epoch as finished for recovery.
        save_checkpoint(
            RECOVERY_PATH,
            model,
            optimizer,
            epoch + 1,
            0,
            train_cases,
            val_cases,
            best_val_dice,
            [],
        )

    print("\nTraining run completed.")
    print("Metrics:", METRICS_PATH)
    print("Recovery checkpoint:", RECOVERY_PATH)
    print("Best checkpoint:", BEST_CHECKPOINT_PATH)
    print(
        "Best validation patch Dice:",
        round(best_val_dice, 4),
    )
    print(
        "Reminder: these patch metrics are preliminary. "
        "Evaluate complete validation volumes before reporting "
        "final segmentation performance."
    )


if __name__ == "__main__":
    main()

"""Generate the train/validation split JSON for BraTS-GLI 2023.

BraTS-GLI 2023 folder structure assumed:
    data_dir/
        BraTS-GLI-00000-000/
            BraTS-GLI-00000-000-t1n.nii.gz
            BraTS-GLI-00000-000-t1c.nii.gz
            BraTS-GLI-00000-000-t2w.nii.gz
            BraTS-GLI-00000-000-t2f.nii.gz
            BraTS-GLI-00000-000-seg.nii.gz
        BraTS-GLI-00001-000/
            ...

Output JSON (saved to data_dir/brats2023_splits.json):
    {
        "training":   [{"image": [...4 paths...], "label": "...seg path..."}, ...],
        "validation": [...]
    }

Usage:
    python src/medseg_brats/data/split_generator.py --data_dir data/BraTS2023_GLI
    python src/medseg_brats/data/split_generator.py --data_dir data/BraTS2023_GLI --val_frac 0.15 --seed 0
"""

import argparse
import json
import random
from pathlib import Path


def generate_splits(
    data_dir: str | Path,
    val_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Scan data_dir for BraTS-GLI cases and build train/val split dict."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case subdirectories found in {data_dir}")

    entries = []
    missing = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        t1n = case_dir / f"{case_id}-t1n.nii.gz"
        t1c = case_dir / f"{case_id}-t1c.nii.gz"
        t2w = case_dir / f"{case_id}-t2w.nii.gz"
        t2f = case_dir / f"{case_id}-t2f.nii.gz"
        seg = case_dir / f"{case_id}-seg.nii.gz"

        paths = [t1n, t1c, t2w, t2f, seg]
        if not all(p.exists() for p in paths):
            missing.append(case_id)
            continue

        entries.append(
            {
                "image": [str(t1n), str(t1c), str(t2w), str(t2f)],
                "label": str(seg),
            }
        )

    if missing:
        print(f"WARNING: {len(missing)} cases skipped due to missing files: {missing[:5]}...")

    if not entries:
        raise RuntimeError("No valid cases found. Check that data_dir contains BraTS-GLI cases.")

    random.seed(seed)
    random.shuffle(entries)

    n_val = max(1, int(len(entries) * val_frac))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    print(f"Total cases: {len(entries)}  |  Train: {len(train_entries)}  |  Val: {len(val_entries)}")
    return {"training": train_entries, "validation": val_entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BraTS-GLI 2023 train/val split JSON.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the BraTS-GLI 2023 directory containing one subfolder per case.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction of cases to use for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to <data_dir>/../brats2023_splits.json.",
    )
    args = parser.parse_args()

    splits = generate_splits(args.data_dir, val_frac=args.val_frac, seed=args.seed)

    output_path = Path(args.output) if args.output else Path(args.data_dir).parent / "brats2023_splits.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved splits to: {output_path}")


if __name__ == "__main__":
    main()

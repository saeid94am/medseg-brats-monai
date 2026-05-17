"""Tests for the BraTS dataset class and split generator.

Test 9 of 10. CPU-only.
"""

from pathlib import Path

import pytest

from medseg_brats.data.dataset import BraTSDataset
from medseg_brats.data.split_generator import generate_splits
from medseg_brats.data.transforms import get_val_transforms


def test_brats_dataset_loading(brats_splits_json):
    """BraTSDataset loads one case and returns correct keys and channel counts.

    Uses val transforms (no crop) so the 100^3 synthetic volume passes through intact.
    cache_rate=0.0 so no disk cache is needed during testing.
    """
    ds = BraTSDataset(
        json_list=brats_splits_json,
        split="training",
        transforms=get_val_transforms(),
        cache_rate=0.0,
        num_workers=0,
    )

    assert len(ds) == 1, f"Expected 1 case, got {len(ds)}"

    sample = ds[0]
    assert "image" in sample, "Sample must contain 'image' key"
    assert "label" in sample, "Sample must contain 'label' key"

    # 4 input modalities stacked into one tensor
    assert (
        sample["image"].shape[0] == 4
    ), f"Image must have 4 channels (modalities), got {sample['image'].shape[0]}"
    # 3 binary region channels (WT, TC, ET) from ConvertToMultiChannelBasedOnBratsClassesd
    assert (
        sample["label"].shape[0] == 3
    ), f"Label must have 3 channels (WT/TC/ET), got {sample['label'].shape[0]}"
    # Spatial dims must match between image and label
    assert (
        sample["image"].shape[1:] == sample["label"].shape[1:]
    ), "Image and label spatial dimensions must match"


def test_brats_dataset_missing_json(tmp_path):
    """BraTSDataset raises FileNotFoundError when the splits JSON doesn't exist."""
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        BraTSDataset(
            json_list=str(tmp_path / "nonexistent.json"),
            split="training",
            transforms=get_val_transforms(),
        )


def test_split_generator(tmp_path, synthetic_brats_case):
    """generate_splits() produces correct training/validation split proportions.

    Uses a synthetic directory structure matching the expected BraTS layout.
    """
    # The fixture already created one case; point generate_splits at its parent
    data_dir = Path(synthetic_brats_case["case_dir"]).parent

    splits = generate_splits(data_dir=data_dir, val_frac=0.2, seed=42)

    assert "training" in splits and "validation" in splits
    total = len(splits["training"]) + len(splits["validation"])
    assert total == 1, f"Expected 1 case total, got {total}"

    # Each entry must have 'image' (list of 4 paths) and 'label' (str)
    for split_name in ("training", "validation"):
        for entry in splits[split_name]:
            assert "image" in entry and "label" in entry
            assert len(entry["image"]) == 4, "Each case must have exactly 4 modality paths"

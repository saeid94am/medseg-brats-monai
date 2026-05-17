"""Tests for the MONAI transform pipelines.

Tests 1–2 of 10. All CPU-only.
"""

import numpy as np
import pytest
import torch
from monai.transforms import ConvertToMultiChannelBasedOnBratsClassesd

from medseg_brats.data.transforms import get_train_transforms, get_val_transforms


def test_brats_label_conversion():
    """ConvertToMultiChannelBasedOnBratsClassesd produces 3 correct binary channels.

    BraTS label mapping:
        WT (ch 0) = labels 1 ∪ 2 ∪ 3
        TC (ch 1) = labels 1 ∪ 3
        ET (ch 2) = label  3
    """
    # Build a (1, 4, 4, 4) label volume with one voxel per class
    label = np.zeros((1, 4, 4, 4), dtype=np.int8)
    label[0, 0, 0, 0] = 1  # NCR  → WT=1, TC=1, ET=0
    label[0, 1, 0, 0] = 2  # SNFH → WT=1, TC=0, ET=0
    label[0, 2, 0, 0] = 3  # ET   → WT=1, TC=1, ET=1
    # label[0, 3, 0, 0] = 0  # background → all zeros (default)

    transform = ConvertToMultiChannelBasedOnBratsClassesd(keys="label")
    result = transform({"label": label})["label"]

    assert result.shape[0] == 3, f"Expected 3 channels, got {result.shape[0]}"

    # WT channel: all non-background voxels
    assert result[0, 0, 0, 0] == 1, "NCR should be in WT"
    assert result[0, 1, 0, 0] == 1, "SNFH should be in WT"
    assert result[0, 2, 0, 0] == 1, "ET should be in WT"
    assert result[0, 3, 0, 0] == 0, "Background should not be in WT"

    # TC channel: NCR + ET only
    assert result[1, 0, 0, 0] == 1, "NCR should be in TC"
    assert result[1, 1, 0, 0] == 0, "SNFH should NOT be in TC"
    assert result[1, 2, 0, 0] == 1, "ET should be in TC"

    # ET channel: ET only
    assert result[2, 0, 0, 0] == 0, "NCR should NOT be in ET"
    assert result[2, 1, 0, 0] == 0, "SNFH should NOT be in ET"
    assert result[2, 2, 0, 0] == 1, "ET voxel should be in ET channel"


def test_train_transform_output_shape(synthetic_brats_case):
    """Full train pipeline on a 100^3 synthetic case → expected output shapes.

    Verifies:
    - Image output: (4, 96, 96, 96) — 4 modalities, 96^3 random crop
    - Label output: (3, 96, 96, 96) — 3 binary channels after BraTS conversion
    - Both are float32 tensors
    """
    sample = {
        "image": synthetic_brats_case["image_paths"],
        "label": synthetic_brats_case["seg_path"],
    }

    transforms = get_train_transforms()
    result = transforms(sample)

    img = result["image"]
    lbl = result["label"]

    assert img.shape == (4, 96, 96, 96), f"Image shape {img.shape} != (4, 96, 96, 96)"
    assert lbl.shape == (3, 96, 96, 96), f"Label shape {lbl.shape} != (3, 96, 96, 96)"
    assert img.dtype == torch.float32, f"Image dtype {img.dtype} != float32"
    assert lbl.dtype == torch.float32, f"Label dtype {lbl.dtype} != float32"
    # Label values must be binary after conversion
    assert set(lbl.unique().tolist()).issubset({0.0, 1.0}), "Label must be binary (0 or 1)"


def test_val_transform_output_shape(synthetic_brats_case):
    """Val pipeline returns the full volume (no cropping) with correct channels."""
    sample = {
        "image": synthetic_brats_case["image_paths"],
        "label": synthetic_brats_case["seg_path"],
    }

    transforms = get_val_transforms()
    result = transforms(sample)

    img = result["image"]
    lbl = result["label"]

    # Full 100^3 volume (no crop in val pipeline)
    assert img.shape[0] == 4, "Val image must have 4 modality channels"
    assert lbl.shape[0] == 3, "Val label must have 3 region channels"
    assert img.shape[1:] == lbl.shape[1:], "Image and label spatial dims must match"

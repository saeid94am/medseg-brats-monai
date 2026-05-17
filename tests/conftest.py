"""Shared pytest fixtures for medseg-brats tests.

All fixtures use CPU-only operations and temporary directories so they
run cleanly in CI without a GPU.
"""

import json

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def synthetic_brats_case(tmp_path):
    """One synthetic BraTS case with volumes large enough for the 96^3 train crop.

    Volume shape: 100x100x100 at 1mm isotropic (affine = eye(4)).
    Modality intensities are positive non-zero so NormalizeIntensityd works.
    Label contains all four class values (0, 1, 2, 3).
    """
    shape = (100, 100, 100)
    affine = np.eye(4)
    case_id = "BraTS-GLI-00000-000"
    case_dir = tmp_path / case_id
    case_dir.mkdir()

    image_paths = []
    rng = np.random.default_rng(seed=0)
    for suffix in ["t1n", "t1c", "t2w", "t2f"]:
        data = rng.random(shape).astype(np.float32) + 0.5  # all positive, non-zero
        p = str(case_dir / f"{case_id}-{suffix}.nii.gz")
        nib.save(nib.Nifti1Image(data, affine), p)
        image_paths.append(p)

    label = np.zeros(shape, dtype=np.uint8)
    label[20:60, 20:60, 20:60] = 1  # NCR  → contributes to WT, TC
    label[30:55, 30:55, 30:55] = 2  # SNFH → contributes to WT only
    label[35:50, 35:50, 35:50] = 3  # ET   → contributes to WT, TC, ET
    seg_path = str(case_dir / f"{case_id}-seg.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), seg_path)

    return {
        "image_paths": image_paths,
        "seg_path": seg_path,
        "case_dir": str(case_dir),
        "shape": shape,
    }


@pytest.fixture
def brats_splits_json(tmp_path, synthetic_brats_case):
    """Minimal splits JSON pointing to the synthetic case (used for both splits)."""
    entry = {
        "image": synthetic_brats_case["image_paths"],
        "label": synthetic_brats_case["seg_path"],
    }
    splits = {"training": [entry], "validation": [entry]}
    json_path = str(tmp_path / "brats2023_splits.json")
    with open(json_path, "w") as f:
        json.dump(splits, f)
    return json_path

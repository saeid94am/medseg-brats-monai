"""MONAI transform pipelines for BraTS-GLI 2023.

Training pipeline applies spatial and intensity augmentations to 96^3 random crops.
Validation pipeline applies only deterministic preprocessing — full volumes are passed
to sliding_window_inference at evaluation time.
"""

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    Spacingd,
)

# Keys used throughout — image is a list of 4 NIfTI paths stacked into one tensor
_KEYS = ["image", "label"]


class ConvertBraTS2023Labelsd(MapTransform):
    """Convert BraTS 2023 integer labels to 3-channel binary [WT, TC, ET].

    BraTS 2023 label convention (differs from BraTS 2018 which used label 4 for ET):
        0 = background
        1 = NCR  (necrotic tumor core)
        2 = SNFH (surrounding non-enhancing FLAIR hyperintensity / edema)
        3 = ET   (GD-enhancing tumor)

    Output channels:
        ch 0 (WT) = labels 1 | 2 | 3   whole tumor
        ch 1 (TC) = labels 1 | 3        tumor core (NCR + ET, excludes edema)
        ch 2 (ET) = label 3             enhancing tumor only
    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)  # (1, H, W, D) → (H, W, D)
            if isinstance(img, torch.Tensor):
                result = torch.stack(
                    [
                        (img == 1) | (img == 2) | (img == 3),
                        (img == 1) | (img == 3),
                        img == 3,
                    ],
                    dim=0,
                ).float()
            else:
                result = np.stack(
                    [
                        ((img == 1) | (img == 2) | (img == 3)).astype(np.float32),
                        ((img == 1) | (img == 3)).astype(np.float32),
                        (img == 3).astype(np.float32),
                    ],
                    axis=0,
                )
            d[key] = result
        return d


def get_train_transforms() -> Compose:
    """Return the full training transform pipeline."""
    # Re-seed numpy before Compose to avoid uint32 overflow on Windows (MONAI bug).
    np.random.seed(42)
    return Compose(
        [
            # --- Loading ---
            # image key receives a list of 4 modality paths; LoadImaged stacks them channel-first
            LoadImaged(keys=_KEYS, image_only=False),
            EnsureChannelFirstd(keys=_KEYS),
            # Convert BraTS 2023 integer labels [0,1,2,3] → 3 binary channels [WT, TC, ET]
            ConvertBraTS2023Labelsd(keys="label"),
            # --- Spatial standardisation ---
            Orientationd(keys=_KEYS, axcodes="RAS"),
            Spacingd(
                keys=_KEYS,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # --- Intensity normalisation (per-modality z-score on non-zero voxels) ---
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # --- Random 96^3 crop (4 GB VRAM constraint; 128^3 would OOM) ---
            RandSpatialCropd(keys=_KEYS, roi_size=[96, 96, 96], random_size=False),
            # --- Spatial augmentation ---
            RandFlipd(keys=_KEYS, prob=0.5, spatial_axis=0),
            RandFlipd(keys=_KEYS, prob=0.5, spatial_axis=1),
            RandFlipd(keys=_KEYS, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=_KEYS, prob=0.5, max_k=3),
            RandZoomd(
                keys=_KEYS,
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.2,
                mode=("trilinear", "nearest"),
            ),
            # --- Intensity augmentation ---
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", prob=0.2, std=0.01),
            RandGaussianSmoothd(
                keys="image",
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                prob=0.2,
            ),
            # --- Final type cast ---
            EnsureTyped(keys=_KEYS, dtype="float32"),
        ]
    )


def get_val_transforms() -> Compose:
    """Return the validation transform pipeline.

    No random augmentations or cropping — full volumes are fed to
    sliding_window_inference at eval time.
    """
    np.random.seed(42)
    return Compose(
        [
            LoadImaged(keys=_KEYS, image_only=False),
            EnsureChannelFirstd(keys=_KEYS),
            ConvertBraTS2023Labelsd(keys="label"),
            Orientationd(keys=_KEYS, axcodes="RAS"),
            Spacingd(
                keys=_KEYS,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=_KEYS, dtype="float32"),
        ]
    )

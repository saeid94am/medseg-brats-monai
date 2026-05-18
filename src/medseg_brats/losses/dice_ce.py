"""Loss functions for BraTS multi-label segmentation.

BraTS has 3 overlapping binary channels (WT/TC/ET can all be 1 for the same
voxel), so losses must use sigmoid — not softmax — per channel.

MONAI's DiceCELoss uses nn.CrossEntropyLoss internally when n_channels > 1,
which assumes mutually-exclusive classes and produces incorrect gradients for
overlapping regions. We combine DiceLoss + BCEWithLogitsLoss directly instead.
"""

import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss, DiceLoss


class _DiceBCELoss(nn.Module):
    """DiceLoss + BCEWithLogitsLoss for multi-label binary segmentation."""

    def __init__(
        self, squared_pred: bool = True, smooth_nr: float = 0.0, smooth_dr: float = 1e-6
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(
            to_onehot_y=False,
            sigmoid=True,
            squared_pred=squared_pred,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        )
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice(input, target) + self.bce(input, target)


def build_loss(name: str, focal_gamma: float = 2.0) -> nn.Module:
    """Return the configured loss function.

    Args:
        name: "dice_ce" or "dice_focal".
        focal_gamma: Focal loss gamma parameter (ignored for dice_ce).
    """
    if name == "dice_ce":
        return _DiceBCELoss()
    if name == "dice_focal":
        return DiceFocalLoss(
            to_onehot_y=False,
            sigmoid=True,
            squared_pred=True,
            gamma=focal_gamma,
            smooth_nr=0.0,
            smooth_dr=1e-6,
        )
    raise ValueError(f"Unknown loss name '{name}'. Choose 'dice_ce' or 'dice_focal'.")

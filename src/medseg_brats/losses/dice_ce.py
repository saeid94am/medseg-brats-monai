"""Loss functions for BraTS multi-label segmentation.

Both losses use sigmoid=True because the output has 3 independent binary
channels (WT, TC, ET) — not a mutually-exclusive softmax over classes.

Switch between them via loss.name in configs/train.yaml without touching
any training code.
"""

import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss


def build_loss(name: str, focal_gamma: float = 2.0) -> nn.Module:
    """Return the configured loss function.

    Args:
        name: "dice_ce" or "dice_focal".
        focal_gamma: Focal loss gamma parameter (ignored for dice_ce).
                     Higher values focus more on hard/small regions (e.g. ET).
    """
    if name == "dice_ce":
        return DiceCELoss(
            to_onehot_y=False,   # label is already multi-channel binary
            sigmoid=True,
            squared_pred=True,   # smoother Dice gradient
            smooth_nr=0.0,
            smooth_dr=1e-6,
        )
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

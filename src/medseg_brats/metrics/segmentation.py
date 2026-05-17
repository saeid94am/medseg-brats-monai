"""Segmentation metrics for BraTS evaluation.

Reports Dice (DSC) and 95th-percentile Hausdorff Distance (HD95) per region:
    - Channel 0: Whole Tumor   (WT)
    - Channel 1: Tumor Core    (TC)
    - Channel 2: Enhancing Tumor (ET)

Both metrics use MONAI's implementations which match the official BraTS
evaluation server behaviour.
"""

from dataclasses import dataclass

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


@dataclass
class MetricBundle:
    """Container for the two MONAI metric objects used during validation."""

    dice: DiceMetric
    hd95: HausdorffDistanceMetric

    def reset(self) -> None:
        self.dice.reset()
        self.hd95.reset()

    def update(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        """Accumulate one batch of predictions and labels."""
        self.dice(y_pred=y_pred, y=y)
        self.hd95(y_pred=y_pred, y=y)

    def aggregate(self) -> dict:
        """Return per-region metric values as a flat dict.

        Returns:
            Dict with keys: dice_wt, dice_tc, dice_et, hd95_wt, hd95_tc,
            hd95_et, mean_dice. Values are Python floats (NaN for cases
            where HD95 is undefined due to empty predictions).
        """
        dice_vals, _ = self.dice.aggregate()  # shape [3]
        hd95_vals, _ = self.hd95.aggregate()  # shape [3]

        return {
            "dice_wt": dice_vals[0].item(),
            "dice_tc": dice_vals[1].item(),
            "dice_et": dice_vals[2].item(),
            "hd95_wt": hd95_vals[0].item(),
            "hd95_tc": hd95_vals[1].item(),
            "hd95_et": hd95_vals[2].item(),
            "mean_dice": dice_vals.nanmean().item(),
        }


def build_metrics() -> MetricBundle:
    """Construct the metric bundle used in both train.py and eval.py."""
    dice = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=True,
    )
    hd95 = HausdorffDistanceMetric(
        include_background=True,
        reduction="mean_batch",
        percentile=95,
        get_not_nans=True,
    )
    return MetricBundle(dice=dice, hd95=hd95)

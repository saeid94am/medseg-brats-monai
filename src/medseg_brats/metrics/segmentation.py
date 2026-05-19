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
    """Container for MONAI metric objects used during validation.

    compute_hd95=False (training): Dice only — HD95 distance transforms on full
    240x240x155 volumes spike CPU RAM and crash on low-memory machines.
    compute_hd95=True (eval.py): full Dice + HD95 for final reported results.
    """

    dice: DiceMetric
    hd95: HausdorffDistanceMetric | None

    def reset(self) -> None:
        self.dice.reset()
        if self.hd95 is not None:
            self.hd95.reset()

    def update(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        self.dice(y_pred=y_pred, y=y)
        if self.hd95 is not None:
            self.hd95(y_pred=y_pred, y=y)

    def aggregate(self) -> dict:
        """Return per-region metric values as a flat dict.

        Keys always present: dice_wt, dice_tc, dice_et, mean_dice.
        Keys present only when compute_hd95=True: hd95_wt, hd95_tc, hd95_et.
        """
        dice_vals, _ = self.dice.aggregate()  # shape [3]

        result = {
            "dice_wt": dice_vals[0].item(),
            "dice_tc": dice_vals[1].item(),
            "dice_et": dice_vals[2].item(),
            "mean_dice": dice_vals.nanmean().item(),
        }

        if self.hd95 is not None:
            hd95_vals, _ = self.hd95.aggregate()  # shape [3]
            # inf arises when GT is empty but pred is non-zero; replace with nan.
            hd95_vals = hd95_vals.float()
            hd95_vals[torch.isinf(hd95_vals)] = float("nan")
            result.update(
                {
                    "hd95_wt": hd95_vals[0].item(),
                    "hd95_tc": hd95_vals[1].item(),
                    "hd95_et": hd95_vals[2].item(),
                }
            )

        return result


def build_metrics(compute_hd95: bool = False) -> MetricBundle:
    """Construct the metric bundle.

    Args:
        compute_hd95: Set True only in eval.py. HD95 requires distance
            transforms on full 3D volumes and is too memory-intensive for
            the training validation loop.
    """
    dice = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=True,
    )
    hd95 = (
        HausdorffDistanceMetric(
            include_background=True,
            reduction="mean_batch",
            percentile=95,
            get_not_nans=True,
        )
        if compute_hd95
        else None
    )
    return MetricBundle(dice=dice, hd95=hd95)

"""Tests for segmentation metrics.

Test 8 of 10. CPU-only.
"""

import torch

from medseg_brats.metrics import build_metrics
from medseg_brats.metrics.segmentation import build_metrics


def test_dice_metric_perfect_overlap():
    """DiceMetric returns 1.0 for identical prediction and ground truth."""
    bundle = build_metrics()

    label = torch.randint(0, 2, (1, 3, 16, 16, 16)).float()
    pred = label.clone()  # perfect match

    bundle.update(y_pred=pred, y=label)
    result = bundle.aggregate()

    assert abs(result["dice_wt"] - 1.0) < 1e-4, f"WT Dice={result['dice_wt']:.4f}, expected 1.0"
    assert abs(result["dice_tc"] - 1.0) < 1e-4, f"TC Dice={result['dice_tc']:.4f}, expected 1.0"
    assert abs(result["dice_et"] - 1.0) < 1e-4, f"ET Dice={result['dice_et']:.4f}, expected 1.0"
    assert abs(result["mean_dice"] - 1.0) < 1e-4


def test_dice_metric_zero_overlap():
    """DiceMetric returns 0.0 when prediction and ground truth are perfectly inverted."""
    bundle = build_metrics()

    # All-ones label, all-zeros prediction → zero overlap
    label = torch.ones(1, 3, 8, 8, 8).float()
    pred = torch.zeros(1, 3, 8, 8, 8).float()

    bundle.update(y_pred=pred, y=label)
    result = bundle.aggregate()

    assert result["dice_wt"] == 0.0, f"WT Dice={result['dice_wt']:.4f}, expected 0.0"
    assert result["dice_tc"] == 0.0, f"TC Dice={result['dice_tc']:.4f}, expected 0.0"
    assert result["dice_et"] == 0.0, f"ET Dice={result['dice_et']:.4f}, expected 0.0"


def test_metric_bundle_reset():
    """MetricBundle.reset() clears accumulated state so metrics start fresh."""
    bundle = build_metrics()

    # First accumulation — perfect prediction
    label = torch.ones(1, 3, 8, 8, 8).float()
    bundle.update(y_pred=label, y=label)
    result_before = bundle.aggregate()

    bundle.reset()

    # Second accumulation — zero overlap
    pred_zeros = torch.zeros(1, 3, 8, 8, 8).float()
    bundle.update(y_pred=pred_zeros, y=label)
    result_after = bundle.aggregate()

    assert result_before["mean_dice"] > result_after["mean_dice"], (
        "After reset, mean Dice with zero-overlap prediction should be lower"
    )

"""Tests for loss functions.

Test 7 of 10. CPU-only.
"""

import pytest
import torch

from medseg_brats.losses import build_loss


def test_dicece_perfect_prediction():
    """DiceCELoss < 0.05 when logits map to near-perfect sigmoid predictions.

    Strategy: create logits of +10 where label=1 (sigmoid → ~1.0) and
    -10 where label=0 (sigmoid → ~0.0). This simulates a model that has
    learned a near-perfect solution.
    """
    torch.manual_seed(0)
    label = torch.randint(0, 2, (1, 3, 8, 8, 8)).float()
    # Large positive logit → sigmoid ≈ 1; large negative → sigmoid ≈ 0
    logits = label * 20.0 - 10.0

    loss_fn = build_loss("dice_ce")
    loss = loss_fn(logits, label)

    assert loss.item() < 0.05, f"Expected near-zero loss for perfect prediction, got {loss.item():.4f}"


def test_dicece_random_prediction_is_higher():
    """DiceCELoss for random logits is substantially higher than for perfect logits."""
    torch.manual_seed(1)
    label = torch.randint(0, 2, (1, 3, 8, 8, 8)).float()

    loss_fn = build_loss("dice_ce")

    random_logits = torch.randn_like(label)
    perfect_logits = label * 20.0 - 10.0

    random_loss = loss_fn(random_logits, label).item()
    perfect_loss = loss_fn(perfect_logits, label).item()

    assert random_loss > perfect_loss, (
        f"Random loss ({random_loss:.4f}) should exceed perfect loss ({perfect_loss:.4f})"
    )


def test_dice_focal_builds():
    """build_loss('dice_focal') returns a callable module without error."""
    loss_fn = build_loss("dice_focal", focal_gamma=2.0)
    assert callable(loss_fn)

    torch.manual_seed(2)
    label = torch.randint(0, 2, (1, 3, 8, 8, 8)).float()
    logits = torch.randn_like(label)
    loss = loss_fn(logits, label)

    assert loss.item() > 0, "Focal loss must be positive for random predictions"


def test_build_loss_invalid_name():
    """build_loss raises ValueError for an unknown loss name."""
    with pytest.raises(ValueError, match="Unknown loss name"):
        build_loss("unknown_loss")

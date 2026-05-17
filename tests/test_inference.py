"""Tests for sliding-window inference.

Test 10 of 10. CPU-only.
"""

import torch
from monai.inferers import sliding_window_inference

from medseg_brats.models.unet3d import build_unet3d

# Reuse the small UNet3D config from test_models for consistency
_UNET3D_CPU_CFG = dict(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=[16, 32, 64],
    strides=[2, 2],
    num_res_units=1,
    dropout=0.0,
)


def test_sliding_window_output_shape():
    """sliding_window_inference output has the same spatial shape as the input.

    Uses a 64^3 volume with 32^3 windows — simulates the full-volume inference
    pattern used at eval time (240^3 volume, 96^3 windows).
    """
    model = build_unet3d(**_UNET3D_CPU_CFG)
    model.eval()

    volume = torch.randn(1, 4, 64, 64, 64)

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=volume,
            roi_size=[32, 32, 32],
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )

    assert output.shape == (1, 3, 64, 64, 64), (
        f"Expected (1, 3, 64, 64, 64), got {output.shape}"
    )


def test_sliding_window_values_are_logits():
    """Output values before sigmoid are unbounded (not clipped to [0,1]).

    Verifies that sliding_window_inference does NOT apply sigmoid internally —
    the caller (eval.py) is responsible for thresholding.
    """
    model = build_unet3d(**_UNET3D_CPU_CFG)
    model.eval()

    volume = torch.randn(1, 4, 32, 32, 32)

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=volume,
            roi_size=[16, 16, 16],
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            mode="constant",
        )

    # Raw logits can be negative — sigmoid not applied
    assert output.min().item() < 0.0 or output.max().item() > 1.0, (
        "Expected raw logits (not sigmoid-activated) from sliding_window_inference"
    )

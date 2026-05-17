"""Tests for model factory functions.

Tests 3–6 of 10. All CPU-only (use small inputs for speed).
GPU tests are marked and skipped automatically in CI.
"""

import sys

import pytest
import torch

from medseg_brats.models.dynunet import build_dynunet, count_parameters
from medseg_brats.models.segresnet import build_segresnet
from medseg_brats.models.unet3d import build_unet3d

# Minimal DynUNet config for CPU tests (3 stages, small filters)
_DYNUNET_CPU_CFG = dict(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size=[[2, 2, 2], [2, 2, 2]],
    filters=[16, 32, 64],
    dropout=0.0,
    deep_supervision=True,
    deep_supr_num=1,
)

# Minimal SegResNet config for CPU tests
_SEGRESNET_CPU_CFG = dict(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    init_filters=8,
    blocks_down=[1, 1, 1, 1],
    blocks_up=[1, 1, 1],
    dropout_prob=0.0,
)

# Minimal UNet3D config for CPU tests
_UNET3D_CPU_CFG = dict(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=[16, 32, 64],
    strides=[2, 2],
    num_res_units=1,
    dropout=0.0,
)

_INPUT = torch.randn(1, 4, 32, 32, 32)  # shared small input for speed


def test_dynunet_forward_cpu():
    """DynUNet forward pass on CPU returns a list (deep supervision) with correct shape."""
    model = build_dynunet(**_DYNUNET_CPU_CFG)
    model.eval()

    with torch.no_grad():
        outputs = model(_INPUT)

    # DynUNet with deep_supervision=True returns a list
    assert isinstance(outputs, (list, tuple)), "DynUNet with deep_supervision must return a list"
    assert outputs[0].shape == (1, 3, 32, 32, 32), (
        f"First output shape {outputs[0].shape} != (1, 3, 32, 32, 32)"
    )
    assert count_parameters(model) > 0


def test_segresnet_forward_cpu():
    """SegResNet forward pass on CPU returns a tensor of correct shape."""
    model = build_segresnet(**_SEGRESNET_CPU_CFG)
    model.eval()

    with torch.no_grad():
        output = model(_INPUT)

    assert output.shape == (1, 3, 32, 32, 32), (
        f"SegResNet output shape {output.shape} != (1, 3, 32, 32, 32)"
    )
    assert count_parameters(model) > 0


def test_unet3d_forward_cpu():
    """3D UNet forward pass on CPU returns a tensor of correct shape."""
    model = build_unet3d(**_UNET3D_CPU_CFG)
    model.eval()

    with torch.no_grad():
        output = model(_INPUT)

    assert output.shape == (1, 3, 32, 32, 32), (
        f"UNet3D output shape {output.shape} != (1, 3, 32, 32, 32)"
    )
    assert count_parameters(model) > 0


def test_model_compile():
    """torch.compile() with backend='eager' runs a forward pass without error.

    backend='eager' works on all platforms without Triton (CPU + GPU).
    Skipped on Windows where torch.compile can be unstable without GPU drivers.
    """
    if sys.platform == "win32" and not torch.cuda.is_available():
        pytest.skip("torch.compile is unreliable on Windows without CUDA")

    model = build_unet3d(**_UNET3D_CPU_CFG)
    compiled = torch.compile(model, backend="eager")
    compiled.eval()

    with torch.no_grad():
        output = compiled(_INPUT)

    assert output.shape == (1, 3, 32, 32, 32)

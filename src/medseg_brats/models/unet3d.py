"""3D U-Net factory — vanilla baseline.

The simplest architecture in this comparison. Expected to underperform DynUNet
but required for the results table (provides a "plain U-Net" reference point
that every BraTS paper includes).

Framework: MONAI UNet with spatial_dims=3 and optional residual units.
"""

import torch.nn as nn
from monai.networks.nets import UNet


def build_unet3d(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    channels: list,
    strides: list,
    num_res_units: int,
    dropout: float,
    **kwargs,
) -> UNet:
    """Instantiate 3D UNet from Hydra config parameters.

    All arguments map 1-to-1 to keys in configs/model/unet3d.yaml.
    **kwargs absorbs extra keys (e.g. 'name', '_target_') added by Hydra.
    """
    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

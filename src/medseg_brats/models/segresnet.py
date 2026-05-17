"""SegResNet factory — compact 3D residual encoder-decoder.

SegResNet is the smallest and fastest model in this project. Train it first
(Week 3) to validate the end-to-end pipeline before committing to DynUNet's
longer training run.

Framework: MONAI (native SegResNet implementation with 3D support).
"""

import torch.nn as nn
from monai.networks.nets import SegResNet


def build_segresnet(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    init_filters: int,
    blocks_down: list,
    blocks_up: list,
    dropout_prob: float,
    **kwargs,
) -> SegResNet:
    """Instantiate SegResNet from Hydra config parameters.

    All arguments map 1-to-1 to keys in configs/model/segresnet.yaml.
    **kwargs absorbs extra keys (e.g. 'name', '_target_') added by Hydra.
    """
    return SegResNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=dropout_prob,
    )


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

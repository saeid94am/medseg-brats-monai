"""DynUNet factory — MONAI's nnU-Net replica.

DynUNet is the primary model for this project. It replicates the nnU-Net
self-configuring architecture: kernel sizes, strides, and feature map counts
are specified explicitly (rather than being auto-derived) so the config is
fully reproducible and version-controlled.

Framework: MONAI (chosen for its production-grade 3D medical imaging support
and native BraTS transform integration — see glossary.md for trade-offs).
"""

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet


def build_dynunet(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: list,
    strides: list,
    upsample_kernel_size: list,
    filters: list,
    dropout: float,
    deep_supervision: bool,
    deep_supr_num: int,
    **kwargs,
) -> DynUNet:
    """Instantiate DynUNet from Hydra config parameters.

    All arguments map 1-to-1 to keys in configs/model/dynunet.yaml.
    **kwargs absorbs extra keys (e.g. 'name', '_target_') added by Hydra.
    """
    return DynUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=filters,
        dropout=dropout,
        deep_supervision=deep_supervision,
        deep_supr_num=deep_supr_num,
    )


def apply_gradient_checkpointing(model: nn.Module) -> None:
    """Wrap DynUNet encoder down-blocks with gradient checkpointing.

    Reduces peak VRAM by ~35% by recomputing encoder activations during
    backward instead of storing them. Adds ~20% training time overhead.
    Only call this when cfg.training.grad_checkpoint is True.

    Uses use_reentrant=False (required for PyTorch 2.x compatibility).
    """
    import torch.utils.checkpoint as cp

    if not hasattr(model, "downsamples"):
        return

    for block in model.downsamples:
        original_forward = block.forward

        def make_checkpointed(fn):
            def checkpointed_forward(*args, **kw):
                return cp.checkpoint(fn, *args, use_reentrant=False, **kw)

            return checkpointed_forward

        block.forward = make_checkpointed(original_forward)


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Visualize model predictions on BraTS validation cases.

Loads N validation cases, runs sliding-window inference, and saves a figure
showing the FLAIR slice with WT/TC/ET overlays side-by-side.

Usage:
    python scripts/visualize_preds.py \
        --checkpoint results/checkpoints/best_segresnet.pth \
        --model segresnet \
        --cases 0 1 2 3 4 \
        --axis 2 \
        --out results/figures/predictions.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medseg_brats.data.transforms import get_val_transforms
from medseg_brats.models.dynunet import build_dynunet
from medseg_brats.models.segresnet import build_segresnet
from medseg_brats.models.unet3d import build_unet3d

# WT=green, TC=yellow, ET=red  (alpha applied when overlaying)
_COLORS = {
    0: np.array([0.0, 1.0, 0.0]),  # WT
    1: np.array([1.0, 1.0, 0.0]),  # TC
    2: np.array([1.0, 0.0, 0.0]),  # ET
}
_REGION_NAMES = ["WT", "TC", "ET"]
_ALPHA = 0.4


def _build_model(name: str, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    configs = {
        "segresnet": dict(
            spatial_dims=3, in_channels=4, out_channels=3,
            init_filters=32, blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1], dropout_prob=0.2,
        ),
        "dynunet": dict(
            spatial_dims=3, in_channels=4, out_channels=3,
            kernel_size=[[3,3,3]]*5, strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
            upsample_kernel_size=[[2,2,2]]*4, filters=[32,64,128,256,320],
            dropout=0.1, deep_supervision=False, deep_supr_num=1,
        ),
        "unet3d": dict(
            spatial_dims=3, in_channels=4, out_channels=3,
            channels=[32, 64, 128, 256], strides=[2, 2, 2],
            num_res_units=2, dropout=0.1,
        ),
    }
    builders = {"segresnet": build_segresnet, "dynunet": build_dynunet, "unet3d": build_unet3d}
    if name not in builders:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(builders)}")

    model = builders[name](**configs[name]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    best_dice = ckpt.get("best_mean_dice", float("nan"))
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded {name} from epoch {epoch} (best mean Dice={best_dice:.4f})")
    return model


def _overlay(flair_slice: np.ndarray, pred_slice: np.ndarray) -> np.ndarray:
    """Blend FLAIR grayscale with coloured prediction mask."""
    # Normalise FLAIR to [0,1]
    lo, hi = flair_slice.min(), flair_slice.max()
    gray = (flair_slice - lo) / (hi - lo + 1e-8)
    rgb = np.stack([gray, gray, gray], axis=-1)

    for ch, color in _COLORS.items():
        mask = pred_slice[ch].astype(bool)
        rgb[mask] = (1 - _ALPHA) * rgb[mask] + _ALPHA * color

    return np.clip(rgb, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--model", default="segresnet", help="Model name")
    parser.add_argument("--cases", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Validation case indices to visualise")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2],
                        help="Slice axis: 0=sagittal, 1=coronal, 2=axial")
    parser.add_argument("--splits", default="data/brats2023_splits.json")
    parser.add_argument("--out", default="results/figures/predictions.png")
    parser.add_argument("--roi", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _build_model(args.model, Path(args.checkpoint), device)
    transforms = get_val_transforms()

    with open(args.splits) as f:
        val_cases = json.load(f)["validation"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(args.cases)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # keep 2-D indexing consistent

    col_titles = ["FLAIR (input)", "Prediction overlay", "Ground truth overlay", "Diff (FP+FN)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, case_idx in enumerate(args.cases):
        if case_idx >= len(val_cases):
            print(f"Case index {case_idx} out of range (only {len(val_cases)} val cases). Skipping.")
            continue

        case = val_cases[case_idx]
        print(f"Processing case {case_idx}: {Path(case['image'][0]).parent.name}")

        sample = {"image": case["image"], "label": case["label"]}
        data = transforms(sample)

        image = data["image"].unsqueeze(0).to(device)   # (1,4,H,W,D)
        label = data["label"].numpy()                    # (3,H,W,D)

        with torch.no_grad():
            output = sliding_window_inference(
                inputs=image,
                roi_size=args.roi,
                sw_batch_size=1,
                predictor=model,
                overlap=args.overlap,
                mode="gaussian",
            )
        pred = (torch.sigmoid(output) > 0.5).squeeze(0).cpu().numpy()  # (3,H,W,D)

        # Pick the middle slice along the chosen axis
        flair = image.squeeze(0)[3].cpu().numpy()  # channel 3 = FLAIR (t2f)
        mid = flair.shape[args.axis] // 2

        def sl(vol):
            slices = [slice(None)] * 3
            slices[args.axis] = mid
            return vol[tuple(slices)] if vol.ndim == 3 else vol[(slice(None),) + tuple(slices)]

        flair_sl = sl(flair)
        pred_sl = sl(pred)
        label_sl = sl(label)

        lo, hi = flair_sl.min(), flair_sl.max()
        gray = (flair_sl - lo) / (hi - lo + 1e-8)

        pred_overlay = _overlay(flair_sl, pred_sl)
        gt_overlay = _overlay(flair_sl, label_sl)

        # Diff: false positives (pred=1, gt=0) in blue; false negatives (pred=0, gt=1) in magenta
        diff_rgb = np.stack([gray, gray, gray], axis=-1)
        fp = (pred_sl.any(axis=0)) & (~label_sl.any(axis=0))
        fn = (~pred_sl.any(axis=0)) & (label_sl.any(axis=0))
        diff_rgb[fp] = [0.2, 0.4, 1.0]   # blue = false positive
        diff_rgb[fn] = [1.0, 0.2, 0.8]   # magenta = false negative

        for col, img in enumerate([gray, pred_overlay, gt_overlay, diff_rgb]):
            ax = axes[row, col]
            cmap = "gray" if col == 0 else None
            ax.imshow(np.rot90(img), cmap=cmap, interpolation="nearest")
            ax.set_ylabel(f"Case {case_idx}", fontsize=9)
            ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green",   alpha=0.7, label="WT (Whole Tumor)"),
        Patch(facecolor="yellow",  alpha=0.7, label="TC (Tumor Core)"),
        Patch(facecolor="red",     alpha=0.7, label="ET (Enhancing Tumor)"),
        Patch(facecolor="#3366ff", alpha=0.7, label="False Positive"),
        Patch(facecolor="#ff33cc", alpha=0.7, label="False Negative"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    axis_names = {0: "sagittal", 1: "coronal", 2: "axial"}
    fig.suptitle(
        f"BraTS 2023 — {args.model} predictions (middle {axis_names[args.axis]} slice)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

"""Evaluation entry point for medseg-brats-monai.

Runs sliding-window inference on the full validation set and writes
per-case Dice and HD95 results to results/metrics/results.csv.

Supports four evaluation modes controlled by config flags:
  ensemble.enabled  — average sigmoid outputs from multiple trained models
  tta.enabled       — average predictions over 4 flip orientations per model

See configs/eval.yaml for usage examples.
"""

import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
import wandb
from monai.inferers import sliding_window_inference
from omegaconf import DictConfig, OmegaConf

from medseg_brats.data.dataset import BraTSDataset, build_loader
from medseg_brats.data.transforms import get_val_transforms
from medseg_brats.metrics import build_metrics
from medseg_brats.models.dynunet import build_dynunet
from medseg_brats.models.segresnet import build_segresnet
from medseg_brats.models.unet3d import build_unet3d

log = logging.getLogger(__name__)

# Architectures must exactly match configs/model/*.yaml and training runs.
_MODEL_CONFIGS: dict = {
    "segresnet": (
        build_segresnet,
        dict(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            init_filters=32,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,
        ),
    ),
    "dynunet": (
        build_dynunet,
        dict(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            kernel_size=[[3, 3, 3]] * 5,
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2]] * 4,
            filters=[32, 64, 128, 256, 320],
            dropout=0.1,
            deep_supervision=True,
            deep_supr_num=2,
        ),
    ),
    "unet3d": (
        build_unet3d,
        dict(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=[32, 64, 128, 256],
            strides=[2, 2, 2],
            num_res_units=2,
            dropout=0.1,
        ),
    ),
}


def _load_model(name: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    if name not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_MODEL_CONFIGS)}")
    builder, config = _MODEL_CONFIGS[name]
    model = builder(**config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    best_dice = ckpt.get("best_mean_dice", float("nan"))
    log.info(f"Loaded {name} from epoch {epoch} (best mean Dice: {best_dice:.4f})")
    return model


def _infer(
    model: torch.nn.Module,
    images: torch.Tensor,
    roi_size: list,
    sw_batch_size: int,
    overlap: float,
    mode: str,
    use_tta: bool,
) -> torch.Tensor:
    """Sliding-window inference with optional 3-axis flip TTA.

    Returns sigmoid probabilities of shape (B, C, H, W, D).
    """
    if not use_tta:
        out = sliding_window_inference(
            images, roi_size, sw_batch_size, model, overlap=overlap, mode=mode
        )
        return torch.sigmoid(out)

    # TTA: original + flip along H (2), W (3), D (4) — flip back before averaging
    preds = []
    for axes in [None, [2], [3], [4]]:
        inp = torch.flip(images, axes) if axes else images
        out = sliding_window_inference(
            inp, roi_size, sw_batch_size, model, overlap=overlap, mode=mode
        )
        if axes:
            out = torch.flip(out, axes)
        preds.append(torch.sigmoid(out))
    return torch.stack(preds).mean(dim=0)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    if not cfg.ensemble.enabled and cfg.inference.checkpoint is None:
        raise ValueError(
            "inference.checkpoint is required for single-model eval.\n"
            "Example: python eval.py model=segresnet "
            "inference.checkpoint=results/checkpoints/best_segresnet.pth\n"
            "Or use ensemble: ensemble.enabled=true"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    val_ds = BraTSDataset(
        json_list=cfg.data.json_list,
        split="validation",
        transforms=get_val_transforms(),
        cache_rate=0.0,
        num_workers=cfg.data.num_workers,
    )
    val_loader = build_loader(val_ds, batch_size=1, shuffle=False, num_workers=0, threaded=False)
    log.info(f"Evaluating on {len(val_ds)} validation cases")

    # ------------------------------------------------------------------ #
    # Model(s)
    # ------------------------------------------------------------------ #
    if cfg.ensemble.enabled:
        model_names = list(cfg.ensemble.models)
        ckpt_paths = list(cfg.ensemble.checkpoints)
        models = [_load_model(n, p, device) for n, p in zip(model_names, ckpt_paths)]
        run_name = f"ensemble_{len(models)}m" + ("_tta" if cfg.tta.enabled else "")
        log.info(f"Ensemble: {model_names}")
    else:
        models = [_load_model(cfg.model.name, cfg.inference.checkpoint, device)]
        run_name = f"eval_{cfg.model.name}" + ("_tta" if cfg.tta.enabled else "")

    log.info(f"TTA: {'enabled (4 flips)' if cfg.tta.enabled else 'disabled'}")

    # ------------------------------------------------------------------ #
    # W&B
    # ------------------------------------------------------------------ #
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=run_name,
        tags=["eval", "phase1"]
        + (list(cfg.ensemble.models) if cfg.ensemble.enabled else [cfg.model.name]),
    )

    # ------------------------------------------------------------------ #
    # Inference + metrics
    # ------------------------------------------------------------------ #
    metrics = build_metrics(compute_hd95=True)
    per_case_rows = []

    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            val_images = val_batch["image"].to(device)
            val_labels = val_batch["label"].to(device)

            # Per-model predictions, then average probabilities across models
            model_preds = [
                _infer(
                    m,
                    val_images,
                    cfg.inference.roi_size,
                    cfg.inference.sw_batch_size,
                    cfg.inference.overlap,
                    cfg.inference.mode,
                    cfg.tta.enabled,
                )
                for m in models
            ]
            ensemble_prob = torch.stack(model_preds).mean(dim=0)
            val_preds = (ensemble_prob > cfg.inference.threshold).float()

            case_metrics = build_metrics(compute_hd95=True)
            case_metrics.update(y_pred=val_preds, y=val_labels)
            case_result = case_metrics.aggregate()

            per_case_rows.append({"case_id": idx, **case_result})
            log.info(
                f"[{idx:04d}] Dice WT={case_result['dice_wt']:.4f} "
                f"TC={case_result['dice_tc']:.4f} ET={case_result['dice_et']:.4f}"
            )

            metrics.update(y_pred=val_preds, y=val_labels)
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    summary = metrics.aggregate()
    log.info("\n=== Evaluation Summary ===")
    log.info(f"  Mode: {run_name}")
    log.info(
        f"  Dice   WT={summary['dice_wt']:.4f}  TC={summary['dice_tc']:.4f}  ET={summary['dice_et']:.4f}"
    )
    log.info(
        f"  HD95   WT={summary['hd95_wt']:.2f}  TC={summary['hd95_tc']:.2f}  ET={summary['hd95_et']:.2f}"
    )
    log.info(f"  Mean Dice: {summary['mean_dice']:.4f}")

    if summary["dice_et"] >= 0.80 and summary["dice_wt"] >= 0.85:
        log.info("Phase 2 threshold met (ET >= 0.80, WT >= 0.85)")
    else:
        log.info("Phase 2 threshold NOT yet met")

    wandb.log({f"eval/{k}": v for k, v in summary.items()})
    wandb.run.summary.update({f"eval/{k}": v for k, v in summary.items()})

    # ------------------------------------------------------------------ #
    # Save per-case CSV
    # ------------------------------------------------------------------ #
    metrics_dir = Path(cfg.output.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / cfg.output.csv_name

    df = pd.DataFrame(per_case_rows)
    df.to_csv(csv_path, index=False)
    log.info(f"Per-case results saved to {csv_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

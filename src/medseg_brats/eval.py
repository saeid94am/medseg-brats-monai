"""Evaluation entry point for medseg-brats-monai.

Runs sliding-window inference on the full validation set and writes
per-case Dice and HD95 results to results/metrics/results.csv.

Usage:
    python src/medseg_brats/eval.py \
        model=dynunet \
        inference.checkpoint=results/checkpoints/best_dynunet.pth
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

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    if cfg.inference.checkpoint is None:
        raise ValueError(
            "inference.checkpoint is required.\n"
            "Example: python eval.py model=dynunet "
            "inference.checkpoint=results/checkpoints/best_dynunet.pth"
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
        cache_rate=1.0,
        num_workers=cfg.data.num_workers,
    )
    val_loader = build_loader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    log.info(f"Evaluating on {len(val_ds)} validation cases")

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = hydra.utils.instantiate(cfg.model).to(device)
    ckpt = torch.load(cfg.inference.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
             f"(best mean Dice: {ckpt.get('best_mean_dice', '?'):.4f})")

    # ------------------------------------------------------------------ #
    # W&B (optional — log eval run separately)
    # ------------------------------------------------------------------ #
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"eval_{cfg.model.name}",
        tags=[cfg.model.name, "eval", "phase1"],
    )

    # ------------------------------------------------------------------ #
    # Inference + metrics
    # ------------------------------------------------------------------ #
    metrics = build_metrics()
    per_case_rows = []

    with torch.no_grad():
        for idx, val_batch in enumerate(val_loader):
            val_images = val_batch["image"].to(device)
            val_labels = val_batch["label"].to(device)

            val_outputs = sliding_window_inference(
                inputs=val_images,
                roi_size=cfg.inference.roi_size,
                sw_batch_size=cfg.inference.sw_batch_size,
                predictor=model,
                overlap=cfg.inference.overlap,
                mode=cfg.inference.mode,
            )
            val_preds = (torch.sigmoid(val_outputs) > cfg.inference.threshold).float()

            # Per-case metrics using a fresh metric object
            case_metrics = build_metrics()
            case_metrics.update(y_pred=val_preds, y=val_labels)
            case_result = case_metrics.aggregate()

            per_case_rows.append({"case_id": idx, **case_result})
            log.info(
                f"[{idx:04d}] Dice WT={case_result['dice_wt']:.4f} "
                f"TC={case_result['dice_tc']:.4f} ET={case_result['dice_et']:.4f}"
            )

            # Accumulate for aggregate
            metrics.update(y_pred=val_preds, y=val_labels)
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Aggregate summary
    # ------------------------------------------------------------------ #
    summary = metrics.aggregate()
    log.info("\n=== Evaluation Summary ===")
    log.info(f"  Dice   WT={summary['dice_wt']:.4f}  TC={summary['dice_tc']:.4f}  ET={summary['dice_et']:.4f}")
    log.info(f"  HD95   WT={summary['hd95_wt']:.2f}  TC={summary['hd95_tc']:.2f}  ET={summary['hd95_et']:.2f}")
    log.info(f"  Mean Dice: {summary['mean_dice']:.4f}")

    # Threshold check (from plan — Phase 2 gate)
    if summary["dice_et"] >= 0.80 and summary["dice_wt"] >= 0.85:
        log.info("✓ Phase 2 threshold met (ET ≥ 0.80, WT ≥ 0.85)")
    else:
        log.info("✗ Phase 2 threshold NOT yet met — continue training or tune hyperparameters")

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

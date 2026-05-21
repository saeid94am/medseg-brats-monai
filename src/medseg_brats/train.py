"""Training entry point for medseg-brats-monai.

Usage:
    # Train DynUNet (default)
    python src/medseg_brats/train.py

    # Train SegResNet with a custom learning rate
    python src/medseg_brats/train.py model=segresnet training.lr=5e-5

    # Debug VRAM usage on first epoch only
    python src/medseg_brats/train.py training.max_epochs=1 debug_memory=true

All hyperparameters live in configs/train.yaml and configs/model/*.yaml.
No values are hardcoded in this file.
"""

import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_

from medseg_brats.data.dataset import BraTSDataset, build_loader
from medseg_brats.data.transforms import get_train_transforms, get_val_transforms
from medseg_brats.losses import build_loss
from medseg_brats.metrics import build_metrics
from medseg_brats.models.dynunet import apply_gradient_checkpointing

log = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    set_determinism(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # must be False for full determinism


def _log_memory(step_label: str) -> None:
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    log.info(f"[{step_label}] VRAM allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # ------------------------------------------------------------------ #
    # 1. Reproducibility
    # ------------------------------------------------------------------ #
    _set_seeds(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # 2. Weights & Biases
    # ------------------------------------------------------------------ #
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{cfg.model.name}_seed{cfg.training.seed}",
        tags=[cfg.model.name, "phase1"],
    )

    # ------------------------------------------------------------------ #
    # 3. Data
    # ------------------------------------------------------------------ #
    train_ds = BraTSDataset(
        json_list=cfg.data.json_list,
        split="training",
        transforms=get_train_transforms(),
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
    )
    val_ds = BraTSDataset(
        json_list=cfg.data.json_list,
        split="validation",
        transforms=get_val_transforms(),
        cache_rate=cfg.data.cache_rate,
        num_workers=cfg.data.num_workers,
    )
    train_loader = build_loader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_loader = build_loader(
        val_ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers, threaded=False
    )
    log.info(f"Train cases: {len(train_ds)} | Val cases: {len(val_ds)}")

    # ------------------------------------------------------------------ #
    # 4. Model
    # ------------------------------------------------------------------ #
    model = hydra.utils.instantiate(cfg.model).to(device)

    if cfg.training.grad_checkpoint:
        apply_gradient_checkpointing(model)
        log.info("Gradient checkpointing enabled")

    if cfg.training.compile:
        model = torch.compile(model, mode="reduce-overhead")
        log.info("torch.compile() enabled")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {cfg.model.name} | Trainable params: {n_params:,}")
    wandb.config.update({"trainable_params": n_params})

    # ------------------------------------------------------------------ #
    # 5. Loss, optimiser, scheduler, AMP scaler
    # ------------------------------------------------------------------ #
    loss_fn = build_loss(cfg.loss.name, focal_gamma=cfg.loss.focal_gamma)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.scheduler.T_0,
        T_mult=cfg.scheduler.T_mult,
        eta_min=cfg.scheduler.eta_min,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)

    # ------------------------------------------------------------------ #
    # 6. Output directories
    # ------------------------------------------------------------------ #
    ckpt_dir = Path(cfg.output.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 7. Metrics
    # ------------------------------------------------------------------ #
    metrics = build_metrics()

    # ------------------------------------------------------------------ #
    # 8. Resume from checkpoint (optional)
    # ------------------------------------------------------------------ #
    best_mean_dice = 0.0
    start_epoch = 0

    resume_path = cfg.training.get("resume_from", None)
    if resume_path:
        resume_path = Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_mean_dice = ckpt.get("best_mean_dice", 0.0)
        start_epoch = ckpt["epoch"] + 1
        log.info(
            f"Resumed from {resume_path} | epoch={ckpt['epoch']} | best_dice={best_mean_dice:.4f}"
        )

    # ------------------------------------------------------------------ #
    # 9. Training loop
    # ------------------------------------------------------------------ #
    grad_accum = cfg.training.grad_accum_steps

    for epoch in range(start_epoch, cfg.training.max_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)  # [B, 4, 96, 96, 96]
            labels = batch["label"].to(device)  # [B, 3, 96, 96, 96]

            with torch.cuda.amp.autocast(enabled=cfg.training.amp):
                outputs = model(images)

                # DynUNet with deep_supervision returns (B, num_heads, C, H, W, D);
                # all other models return (B, C, H, W, D).
                if outputs.ndim == 6:
                    n_heads = outputs.shape[1]
                    weights = [1.0, 0.5, 0.25][:n_heads]
                    loss = sum(loss_fn(outputs[:, i], labels) * w for i, w in enumerate(weights))
                else:
                    loss = loss_fn(outputs, labels)

                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum

            if cfg.debug_memory and step == 0 and epoch == 0:
                _log_memory("epoch=0 step=0 post-backward")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({"train/loss": avg_loss, "train/lr": current_lr, "epoch": epoch})
        log.info(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | lr={current_lr:.2e}")

        # Save a rolling "last" checkpoint every epoch so resume never loses
        # more than one epoch of training regardless of val_interval.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_mean_dice": best_mean_dice,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_dir / f"last_{cfg.model.name}.pth",
        )

        # -------------------------------------------------------------- #
        # 9. Validation
        # -------------------------------------------------------------- #
        if (epoch + 1) % cfg.training.val_interval == 0:
            model.eval()
            metrics.reset()

            with torch.no_grad():
                for val_batch in val_loader:
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
                    val_preds = (torch.sigmoid(val_outputs) > 0.5).float()
                    metrics.update(y_pred=val_preds, y=val_labels)

            torch.cuda.empty_cache()
            result = metrics.aggregate()

            wandb.log({f"val/{k}": v for k, v in result.items()} | {"epoch": epoch})
            log.info(
                f"  Val | Dice WT={result['dice_wt']:.4f} "
                f"TC={result['dice_tc']:.4f} "
                f"ET={result['dice_et']:.4f} | mean={result['mean_dice']:.4f}"
            )

            # ---------------------------------------------------------- #
            # 10. Checkpoint
            # ---------------------------------------------------------- #
            if result["mean_dice"] > best_mean_dice:
                best_mean_dice = result["mean_dice"]
                ckpt_path = ckpt_dir / f"best_{cfg.model.name}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_mean_dice": best_mean_dice,
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_path,
                )
                log.info(f"  New best mean Dice={best_mean_dice:.4f} -> saved {ckpt_path}")
                wandb.run.summary["best_mean_dice"] = best_mean_dice
                wandb.run.summary["best_epoch"] = epoch

    # ------------------------------------------------------------------ #
    # 11. Final summary
    # ------------------------------------------------------------------ #
    log.info(f"Training complete. Best mean Dice: {best_mean_dice:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()

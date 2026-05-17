#!/usr/bin/env bash
# Reproduce all Phase 1 experiments end-to-end.
#
# Usage:
#   bash scripts/run_experiments.sh              # all three models
#   bash scripts/run_experiments.sh segresnet    # single model (faster sanity check)
#
# Assumes:
#   - pip install -e ".[dev]" has been run
#   - data/brats2023_splits.json exists (run download_data.sh first)
#   - WANDB_ENTITY is set (or wandb.entity is filled in configs/train.yaml)

set -euo pipefail

MODELS="${1:-segresnet dynunet unet3d}"
RESULTS_DIR="results/metrics"

echo "=== medseg-brats-monai: Phase 1 Experiments ==="
echo "Models: $MODELS"
echo ""

for MODEL in $MODELS; do
    echo "--- Training: $MODEL ---"
    python src/medseg_brats/train.py model="$MODEL"

    echo "--- Evaluating: $MODEL ---"
    python src/medseg_brats/eval.py \
        model="$MODEL" \
        inference.checkpoint="results/checkpoints/best_${MODEL}.pth" \
        output.csv_name="${MODEL}_results.csv"

    echo "Finished $MODEL"
    echo ""
done

echo "=== All experiments complete ==="
echo "Results saved to $RESULTS_DIR/"
ls "$RESULTS_DIR/"

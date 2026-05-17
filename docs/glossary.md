# Technical Glossary

Reference for non-trivial terms used in this project. Covers metrics, loss functions, model components, training mechanics, data pipeline, and infrastructure.

---

## Metrics

**Dice Similarity Coefficient (DSC / Dice score)**
Measures overlap between predicted segmentation and ground truth. `DSC = 2|A∩B| / (|A|+|B|)`. Range 0–1; 1 is perfect. Used per region (WT, TC, ET). This is the primary metric for BraTS leaderboards.

**HD95 (95th-percentile Hausdorff Distance)**
Measures how far apart the predicted and true surface boundaries are, in mm, at the 95th percentile (ignores the worst 5% outliers). Lower is better. A model with good Dice but bad HD95 has correct coverage but jagged, inaccurate edges.

**WT / TC / ET (BraTS tumor sub-regions)**
- **WT** = Whole Tumor = labels 1+2+3 (everything non-background)
- **TC** = Tumor Core = labels 1+3 (necrotic core + enhancing tumor)
- **ET** = Enhancing Tumor = label 3 only (the most clinically relevant region; smallest and hardest to segment)

These are derived from a single 4-label segmentation map via `ConvertToMultiChannelBasedOnBratsClassesd`.

---

## Loss Functions

**DiceCELoss**
Combines Dice loss (overlap-based) + Cross-Entropy loss (pixel-wise). Dice alone is unstable for small regions because its gradient vanishes when overlap is near zero; CE provides stable gradients throughout training. The `sigmoid=True` flag applies sigmoid before computing Dice — used for multi-channel binary outputs instead of softmax.

**DiceFocalLoss**
Replaces CE with Focal loss. Focal loss has a `gamma` parameter (default 2.0) that down-weights easy background voxels so the model focuses on hard positives (small tumor regions like ET). Use this if ET Dice stagnates below 0.5 after 100 epochs — switch via `loss.name: dice_focal` in `configs/train.yaml`.

**deep_supervision**
DynUNet returns predictions at multiple decoder levels, not just the final output. Each intermediate prediction receives a weighted loss (weights: `[1.0, 0.5, 0.25]`). Forces the network to build meaningful representations at all scales, which stabilizes early training. Only active during training; at inference only the final decoder output is used.

---

## Models

**DynUNet**
MONAI's re-implementation of nnU-Net — the method that automatically configures U-Net architecture (kernel sizes, strides, feature map sizes) based on patch size and voxel spacing. It is the primary model here because it represents the state-of-the-art CNN approach for 3D segmentation and is the mandatory baseline to beat in any BraTS publication.

**SegResNet**
A compact 3D ResNet-based encoder-decoder. Fewer parameters than DynUNet, faster to train per epoch. Recommended to train first (Week 3) to verify the pipeline before committing to DynUNet's longer training run.

**UNet3D**
MONAI's standard 3D U-Net implementation. Serves as the vanilla baseline — the simplest architecture in the comparison. Performance is expected to be lower than DynUNet but its results are required for the results table.

**SwinUNETR** *(not implemented — 4 GB VRAM constraint)*
Transformer-based segmentation model using a Swin Transformer encoder. Skipped because self-attention scales quadratically with the number of 3D patch tokens, exceeding the 4 GB memory budget at any reasonable patch size.

---

## Training Mechanics

**Mixed Precision / AMP (Automatic Mixed Precision)**
Runs forward/backward passes in FP16 (half-precision, 2 bytes/number) instead of FP32 (4 bytes). Roughly halves VRAM usage and speeds up computation on NVIDIA GPUs with Tensor Cores. `GradScaler` compensates for the reduced numerical range of FP16 by scaling gradients before/after the backward pass.

**GradScaler**
The AMP companion object. Prevents "gradient underflow" — very small FP16 gradients rounding to zero and stopping learning. It multiplies the loss by a large scale factor before `.backward()`, then divides the gradients back before the optimizer step. Automatically reduces the scale factor if NaN/Inf gradients are detected (an "overflow" event).

**Gradient Accumulation (`grad_accum_steps=4`)**
Instead of updating model weights every step, gradients accumulate over 4 steps before an optimizer update. With `batch_size=1` and `grad_accum_steps=4`, you get the same effective batch size of 4 without needing 4× more VRAM. The loss is divided by `grad_accum_steps` before `.backward()` so the gradient magnitude stays correct. **Important:** `optimizer.zero_grad()` is called once before the batch loop, not inside it.

**Gradient Checkpointing**
Trades compute for memory. Normally, all intermediate layer activations are stored in VRAM during the forward pass (needed for computing gradients during backprop). Gradient checkpointing discards them and recomputes them on-the-fly during backward. Saves ~30–40% VRAM at the cost of ~20% slower training. Enabled via `training.grad_checkpoint: true` in the config.

**`torch.compile()`**
PyTorch 2.x feature that JIT-compiles the model graph into optimized native code. Can give 10–30% speedup with no code changes. On Windows it requires the Triton compiler (often unavailable), so it is a config flag (`training.compile: false` by default). On Linux CI it uses `backend="eager"` which works without Triton.

**CosineAnnealingWarmRestarts**
Learning rate scheduler. LR decays from the initial value down to `eta_min` following a cosine curve over `T_0` epochs, then "restarts" (jumps back to the initial LR). Warm restarts help the optimizer escape local minima. Controlled by `scheduler.T_0` in the config.

**AdamW**
Adam optimizer with *decoupled* weight decay (L2 regularization applied directly to weights, bypassing the gradient). The standard Adam optimizer applied weight decay through the gradient update, which is mathematically incorrect — AdamW fixes this. Default optimizer for fine-tuning and medical image segmentation.

**gradient clipping (`max_norm=1.0`)**
Scales the gradient vector down if its L2 norm exceeds `max_norm`. Prevents "exploding gradients" — very large gradient steps that destabilize training. Called via `torch.nn.utils.clip_grad_norm_()` after `scaler.unscale_()` (so norms are measured in FP32 scale, not scaled FP16).

---

## Data Pipeline

**MONAI `CacheDataset`**
Pre-loads a fraction (`cache_rate`) of the dataset into CPU RAM after applying the deterministic transforms (loading, spacing, orientation, normalization). Subsequent epochs skip disk I/O and apply only random augmentations to the cached tensors. Essential for 3D NIfTI data where I/O is slow. Set `cache_rate=0.25` to cache 25% of training cases.

**`ThreadDataLoader`**
MONAI's replacement for PyTorch's `DataLoader`. Uses threads instead of sub-processes for data loading. On Windows, PyTorch's multiprocess DataLoader with `num_workers>0` can deadlock because Windows lacks `os.fork()`. `ThreadDataLoader` avoids this entirely.

**`sliding_window_inference`**
At inference, a full 240×240×155 BraTS volume is too large to process in one forward pass on 4 GB VRAM. This function slides a 96×96×96 window across the volume with 50% overlap, runs the model on each window, then stitches predictions together. `mode="gaussian"` weights each window's contribution by a Gaussian kernel (center weighted higher than edges), significantly reducing boundary artifacts compared to `mode="constant"`.

**`ConvertToMultiChannelBasedOnBratsClassesd`**
MONAI built-in transform. BraTS segmentation files store labels as a single channel with integer values 0/1/2/3. This transform converts them to 3 binary channels: channel 0 = WT (labels 1∪2∪3), channel 1 = TC (labels 1∪3), channel 2 = ET (label 3 only). Required because `DiceCELoss` with `sigmoid=True` operates independently per channel.

**`NormalizeIntensityd(nonzero=True, channel_wise=True)`**
Z-score normalization applied per modality (per channel) using only non-zero voxels. The `nonzero=True` flag is critical — BraTS images are skull-stripped so all background voxels are zero. Including zeros in the mean/std would corrupt the normalization statistics.

**`Orientationd(axcodes="RAS")`**
Standardizes MRI orientation to Right-Anterior-Superior coordinate frame. Different scanners output volumes in different orientations. Without this, two cases from different scanners could have their axes in different orders, causing the model to see anatomically inconsistent inputs.

**`Spacingd(pixdim=(1,1,1))`**
Resamples volumes so each voxel represents exactly 1mm × 1mm × 1mm. BraTS-GLI 2023 is already at 1mm isotropic, so this acts as a safety check and is important if you later mix in other datasets with different spacings.

---

## Infrastructure

**Hydra + OmegaConf**
Hydra is a config management framework. All hyperparameters live in YAML files instead of Python code. The `@hydra.main` decorator loads the YAML and injects it as a structured `DictConfig` object. Any config value can be overridden on the command line: `python train.py training.lr=1e-3 model=segresnet`. OmegaConf is the underlying config library that Hydra uses.

**`_target_` in YAML config**
Special Hydra key. `_target_: medseg_brats.models.dynunet.build_dynunet` tells `hydra.utils.instantiate(cfg.model)` to import and call `build_dynunet` with all sibling YAML keys as keyword arguments. This is how model configs wire to Python factory functions without any if/else in training code.

**pre-commit hooks**
Scripts that run automatically before every `git commit`. Configured in `.pre-commit-config.yaml`. Here they run `ruff` (fast Python linter + formatter), `black` (auto-formatter), `isort` (import sorter), and `nbstripout` (strips notebook output to keep diffs clean). A commit is blocked if any hook modifies or errors on a file — you then re-stage and commit again.

**Weights & Biases (W&B)**
Cloud experiment tracking platform. During training, logs metrics (loss, Dice, HD95 per epoch), the full config, system stats (GPU utilization, memory), and sample prediction images to a web dashboard. Lets you compare runs across models and hyperparameters. Free for personal use. Set `WANDB_ENTITY` environment variable with your W&B username.

**Synapse CLI (`synapseclient`)**
Python client for synapse.org — the platform hosting BraTS 2023 data. Requires a free Synapse account and challenge registration. `scripts/download_data.sh` uses it to automate downloading once credentials are configured.

**MONAI Bundle**
Standardized packaging format for MONAI models. Configs in `configs/bundle/` describe the model architecture, inference settings, and metadata so the model can be loaded and served with a single `monai.bundle run` command. Required for Phase 1 deliverables and is good practice for reproducibility.

---

## BraTS-GLI 2023 Data Format

Each case folder (`BraTS-GLI-XXXXX-XXX/`) contains:
| File suffix | Modality | Description |
|---|---|---|
| `-t1n.nii.gz` | T1 native | Pre-contrast T1-weighted MRI |
| `-t1c.nii.gz` | T1 contrast | Post-contrast T1 (enhances tumor boundary) |
| `-t2w.nii.gz` | T2-weighted | Shows edema as bright regions |
| `-t2f.nii.gz` | T2-FLAIR | Fluid-suppressed T2; best for edema extent |
| `-seg.nii.gz` | Segmentation | Labels: 0=background, 1=NCR, 2=SNFH, 3=ET |

All volumes are 240×240×155 voxels at 1mm isotropic resolution, skull-stripped (background = 0).

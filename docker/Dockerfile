FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

LABEL maintainer="saeed.amiinii@gmail.com"
LABEL description="3D Brain Tumor Segmentation — BraTS-GLI 2023 (Phase 1)"

WORKDIR /workspace

# System dependencies for nibabel / SimpleITK / NIfTI I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (data/ and results/ are excluded via .dockerignore)
COPY pyproject.toml LICENSE ./
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/

# Upgrade pip and install hatchling (build backend for the package)
RUN pip install --upgrade pip hatchling

# Install only training-critical dependencies (skip demo/CI tools that conflict
# with the existing pytorch base image environment)
RUN pip install --no-cache-dir \
    monai==1.3.0 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    wandb==0.16.6 \
    nibabel==5.2.1 \
    SimpleITK==2.3.1 \
    scipy==1.13.0 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    matplotlib==3.8.4 \
    einops==0.7.0 \
    rich==13.7.1 \
    tqdm==4.66.2

# Install the package itself without re-resolving deps (already installed above).
# Non-editable install is appropriate for a container image.
RUN pip install --no-cache-dir --no-deps .

# Default command: verify installation then show usage
CMD ["python", "-c", \
    "import medseg_brats; import monai; import torch; \
     print(f'medseg_brats {medseg_brats.__version__} | MONAI {monai.__version__} | PyTorch {torch.__version__}'); \
     print('GPU available:', torch.cuda.is_available()); \
     print('Run: python src/medseg_brats/train.py')"]

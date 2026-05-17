FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

LABEL maintainer="saeed.amiinii@gmail.com"
LABEL description="3D Brain Tumor Segmentation — BraTS-GLI 2023 (Phase 1)"

WORKDIR /workspace

# System dependencies for nibabel / SimpleITK / NIfTI I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (data/ and results/ are excluded via .dockerignore)
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/

# Install package in editable mode with dev extras
RUN pip install --no-cache-dir -e ".[dev]"

# Default command: verify installation then show usage
CMD ["python", "-c", \
    "import medseg_brats; import monai; import torch; \
     print(f'medseg_brats {medseg_brats.__version__} | MONAI {monai.__version__} | PyTorch {torch.__version__}'); \
     print('GPU available:', torch.cuda.is_available()); \
     print('Run: python src/medseg_brats/train.py')"]

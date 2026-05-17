"""Streamlit demo for medseg-brats-monai.

Upload four BraTS NIfTI modalities → get WT/TC/ET segmentation overlay.

Usage:
    streamlit run demo/app.py
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="BraTS Tumor Segmentation Demo",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 3D Brain Tumor Segmentation — BraTS-GLI 2023")
st.caption(
    "**Research tool only — not validated for clinical use. "
    "Not a medical device. Do not use for diagnosis or treatment decisions.**"
)
st.divider()

# ------------------------------------------------------------------ #
# Sidebar — model selection
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("Model Settings")

    model_name = st.selectbox(
        "Architecture",
        options=["dynunet", "segresnet", "unet3d"],
        index=0,
    )

    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=f"results/checkpoints/best_{model_name}.pth",
        help="Relative or absolute path to a .pth checkpoint file.",
    )

    device_choice = st.radio("Device", options=["cuda", "cpu"], index=0)
    device = torch.device(device_choice if torch.cuda.is_available() else "cpu")
    st.info(f"Running on: **{device}**")

    threshold = st.slider("Sigmoid threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    st.divider()
    st.markdown(
        "**Region colours**\n"
        "- 🟢 Green = Whole Tumor (WT)\n"
        "- 🟡 Yellow = Tumor Core (TC)\n"
        "- 🔴 Red = Enhancing Tumor (ET)"
    )

# ------------------------------------------------------------------ #
# Model loader (cached so the checkpoint is only read once per session)
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="Loading model checkpoint…")
def load_model(name: str, ckpt_path: str, dev: str):
    """Load model from checkpoint. Cached by (name, ckpt_path, device)."""
    import hydra
    from omegaconf import OmegaConf

    # Minimal configs to instantiate each model without Hydra's full config system
    from medseg_brats.models.dynunet import build_dynunet
    from medseg_brats.models.segresnet import build_segresnet
    from medseg_brats.models.unet3d import build_unet3d

    if name == "dynunet":
        model = build_dynunet(
            spatial_dims=3, in_channels=4, out_channels=3,
            kernel_size=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
            strides=[[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
            upsample_kernel_size=[[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
            filters=[32, 64, 128, 256, 320],
            dropout=0.0, deep_supervision=False, deep_supr_num=1,
        )
    elif name == "segresnet":
        model = build_segresnet(
            spatial_dims=3, in_channels=4, out_channels=3,
            init_filters=32, blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1],
            dropout_prob=0.0,
        )
    else:
        model = build_unet3d(
            spatial_dims=3, in_channels=4, out_channels=3,
            channels=[32, 64, 128, 256], strides=[2, 2, 2],
            num_res_units=2, dropout=0.0,
        )

    device_obj = torch.device(dev)
    ckpt = torch.load(ckpt_path, map_location=device_obj)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return model


# ------------------------------------------------------------------ #
# Inference transform (no augmentation, no crop)
# ------------------------------------------------------------------ #
_INFER_TRANSFORMS = Compose([
    LoadImaged(keys=["image"], image_only=False),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"], dtype="float32"),
])


@st.cache_data(show_spinner="Preprocessing volume…", max_entries=3)
def preprocess(t1n_bytes, t1c_bytes, t2w_bytes, t2f_bytes):
    """Save uploads to temp files and apply inference transforms. Cached by file content."""
    tmp = tempfile.mkdtemp()
    paths = []
    for name, data in zip(["t1n", "t1c", "t2w", "t2f"], [t1n_bytes, t1c_bytes, t2w_bytes, t2f_bytes]):
        p = Path(tmp) / f"{name}.nii.gz"
        p.write_bytes(data)
        paths.append(str(p))

    result = _INFER_TRANSFORMS({"image": paths})
    return result["image"].numpy()  # (4, H, W, D)


def run_inference(image_np: np.ndarray, model, threshold: float, dev: torch.device) -> np.ndarray:
    """Run sliding-window inference and return binary prediction (3, H, W, D)."""
    tensor = torch.from_numpy(image_np).unsqueeze(0).to(dev)  # (1, 4, H, W, D)
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=tensor,
            roi_size=[96, 96, 96],
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )
    pred = (torch.sigmoid(output) > threshold).float()
    return pred[0].cpu().numpy()  # (3, H, W, D)


def make_overlay_figure(flair: np.ndarray, pred: np.ndarray, axis: int, slice_idx: int) -> plt.Figure:
    """Overlay WT/TC/ET on a FLAIR slice along the given axis."""
    sl = [slice(None)] * 3
    sl[axis] = slice_idx
    sl = tuple(sl)

    bg = flair[sl]
    wt = pred[0][sl]
    tc = pred[1][sl]
    et = pred[2][sl]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(bg.T, cmap="gray", origin="lower", interpolation="nearest")
    ax.imshow(np.ma.masked_where(wt.T == 0, wt.T), cmap="Greens", alpha=0.4, origin="lower", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(tc.T == 0, tc.T), cmap="YlOrBr", alpha=0.4, origin="lower", vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(et.T == 0, et.T), cmap="Reds",   alpha=0.6, origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ------------------------------------------------------------------ #
# File uploaders
# ------------------------------------------------------------------ #
st.subheader("Step 1 — Upload four MRI modalities")
col1, col2, col3, col4 = st.columns(4)
with col1:
    t1n_file = st.file_uploader("T1 native (t1n)", type=["gz", "nii"], key="t1n")
with col2:
    t1c_file = st.file_uploader("T1 contrast (t1c)", type=["gz", "nii"], key="t1c")
with col3:
    t2w_file = st.file_uploader("T2-weighted (t2w)", type=["gz", "nii"], key="t2w")
with col4:
    t2f_file = st.file_uploader("T2-FLAIR (t2f)", type=["gz", "nii"], key="t2f")

all_uploaded = all([t1n_file, t1c_file, t2w_file, t2f_file])

# ------------------------------------------------------------------ #
# Run segmentation
# ------------------------------------------------------------------ #
st.subheader("Step 2 — Run segmentation")

if st.button("▶ Run Segmentation", disabled=not all_uploaded, type="primary"):
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        st.error(f"Checkpoint not found: `{checkpoint_path}`\n\nTrain a model first with `python src/medseg_brats/train.py`.")
        st.stop()

    with st.spinner("Loading model…"):
        model = load_model(model_name, str(ckpt), str(device))

    with st.spinner("Preprocessing…"):
        image_np = preprocess(
            t1n_file.read(), t1c_file.read(), t2w_file.read(), t2f_file.read()
        )

    with st.spinner("Running inference (this may take a few minutes on CPU)…"):
        pred_np = run_inference(image_np, model, threshold, device)

    st.session_state["image_np"] = image_np
    st.session_state["pred_np"] = pred_np
    st.success("Segmentation complete!")

# ------------------------------------------------------------------ #
# Visualisation
# ------------------------------------------------------------------ #
if "image_np" in st.session_state:
    image_np = st.session_state["image_np"]
    pred_np  = st.session_state["pred_np"]
    flair    = image_np[3]  # T2-FLAIR is the 4th modality (index 3)
    H, W, D  = flair.shape

    st.subheader("Step 3 — Explore slices")

    axis_labels = {"Axial (Z)": 2, "Coronal (Y)": 1, "Sagittal (X)": 0}
    axis_col, _ = st.columns([1, 3])
    with axis_col:
        axis_name = st.selectbox("Plane", list(axis_labels.keys()))
    axis = axis_labels[axis_name]
    max_slice = [H, W, D][axis] - 1

    slice_idx = st.slider("Slice", min_value=0, max_value=max_slice, value=max_slice // 2)

    fig = make_overlay_figure(flair, pred_np, axis, slice_idx)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # Summary statistics
    st.subheader("Prediction summary")
    total_voxels = H * W * D
    c1, c2, c3 = st.columns(3)
    c1.metric("Whole Tumor (WT)", f"{pred_np[0].sum() / total_voxels * 100:.1f}% of volume")
    c2.metric("Tumor Core (TC)",  f"{pred_np[1].sum() / total_voxels * 100:.1f}% of volume")
    c3.metric("Enhancing Tumor (ET)", f"{pred_np[2].sum() / total_voxels * 100:.1f}% of volume")

# ------------------------------------------------------------------ #
# Persistent disclaimer
# ------------------------------------------------------------------ #
st.divider()
st.warning(
    "⚠️ **Disclaimer:** This tool is for research and educational purposes only. "
    "It is not a validated medical device and must not be used for clinical diagnosis, "
    "treatment planning, or any patient care decisions. "
    "Model outputs may be inaccurate and have not been reviewed by medical professionals."
)

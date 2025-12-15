#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_fig1_mast3r_baseline.py

Visualize vanilla MASt3R / DUSt3R baseline (no UQ head):
- Input RGB
- Baseline depth from pts3d z-channel
- Original confidence map

This is the "pure MASt3R baseline" figure to compare against our Geo UQ (Option A).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mast3r.model import AsymmetricMASt3R
from dust3r.datasets.scannetpp import ScanNetpp
from dust3r.utils.image import rgb as denorm_rgb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------- 1) Load vanilla MASt3R / DUSt3R baseline model ----------
def load_mast3r_baseline(ckpt_path: str):
    """
    Load a geometry-only baseline model (no UQ head).
    We use head_type='dpt' so the downstream head is the vanilla DPT head:
      - outputs pts3d and conf.
    """
    model = AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(224, 224),
        head_type="dpt",           # *** vanilla DPT head, no evidential UQ ***
        output_mode="pts3d",
        depth_mode=("exp", float("-inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
        dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)  # allow missing UQ-related keys
    model.to(device)
    model.eval()
    return model


# ---------- 2) Take one ScanNet++ sample ----------
def get_sample(dataset_root: str, idx: int = 0):
    """
    Load one pair (view1, view2) from ScanNet++ and wrap tensors into batch dim.
    """
    ds = ScanNetpp(split="train", ROOT=dataset_root, resolution=224, aug_crop=0)
    view1, view2 = ds[idx]

    def to_batched(view):
        out = {}
        for k, v in view.items():
            if k == "true_shape":
                out[k] = torch.as_tensor(v, device=device).unsqueeze(0)
            elif torch.is_tensor(v):
                out[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            else:
                out[k] = v
        return out

    return to_batched(view1), to_batched(view2)


# ---------- 3) Make the 1×3 figure: RGB / depth(z) / conf ----------
def viz_mast3r_baseline(model, sample, out_path: str):
    """
    Plot:
      [Input RGB]  [Baseline depth (from pts3d z)]
      [Original confidence heatmap]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    view1, view2 = sample
    with torch.no_grad():
        pred1, _ = model(view1, view2)

    print("pred1 keys:", pred1.keys())

    # 1) RGB (de-normalized)
    rgb_img = denorm_rgb(view1["img"][0])

    # 2) Baseline depth from pts3d z-axis
    if "pts3d" not in pred1:
        raise RuntimeError(f"'pts3d' not in prediction dict. Got keys: {pred1.keys()}")
    pts3d = pred1["pts3d"][0].detach().cpu().numpy()
    # Handle both channel-first (3,H,W) and channel-last (H,W,3) layouts
    if pts3d.shape[0] == 3:          # [3, H, W]
        depth_z = pts3d[2]
    elif pts3d.shape[-1] == 3:       # [H, W, 3]
        depth_z = pts3d[..., 2]
    else:
        raise RuntimeError(f"Unexpected pts3d shape: {pts3d.shape}")

    # Avoid extreme outliers messing up the colormap: percentile clipping
    d_min, d_max = np.percentile(depth_z, [5, 95])
    depth_vis = np.clip(depth_z, d_min, d_max)

    # 3) Original confidence map
    conf = pred1.get("conf", None)
    if conf is None:
        raise RuntimeError(f"'conf' not in prediction dict. Got keys: {pred1.keys()}")

    # conf can be (B,H,W) or (B,1,H,W)
    if conf.ndim == 4:
        conf_vis = conf[0, 0].detach().cpu().numpy()
    elif conf.ndim == 3:
        conf_vis = conf[0].detach().cpu().numpy()
    else:
        raise RuntimeError(f"Unexpected conf shape: {conf.shape}")

    print(f"depth_z range : {depth_z.min():.3f} .. {depth_z.max():.3f}")
    print(f"conf range    : {conf_vis.min():.3f} .. {conf_vis.max():.3f}")

    # ---- Plot 1×3 ----
    plt.figure(figsize=(12, 4))

    # (1) Input RGB
    plt.subplot(1, 3, 1)
    plt.title("Input RGB")
    plt.imshow(rgb_img)
    plt.axis("off")

    # (2) Baseline depth from z
    plt.subplot(1, 3, 2)
    plt.title("Baseline depth (z from pts3d)")
    im = plt.imshow(depth_vis, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")

    # (3) Original confidence
    plt.subplot(1, 3, 3)
    plt.title("Original confidence (MASt3R)")
    im = plt.imshow(conf_vis, cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("saved vanilla baseline fig to", out_path)


if __name__ == "__main__":
    # ----- TODO: customize these two paths -----
    ckpt_path = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   # vanilla geometry ckpt
    data_root = "/workspace/data/scannetpp_processed_cut3r"               # your ScanNet++ root
    out_file = "viz_icml/fig0_mast3r_vanilla_baseline.png"

    model = load_mast3r_baseline(ckpt_path)
    sample = get_sample(data_root, idx=0)  # you can change idx to try other views
    viz_mast3r_baseline(model, sample, out_file)

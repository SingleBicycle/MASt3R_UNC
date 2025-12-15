#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple visualization for:
- S0: geometry sanity check (no UQ)
- S1: 3D UQ-only (xyz_var risk map)

Usage examples:

S0:

PYTHONPATH=".:dust3r/dust3r" python tools/vis_scannetpp_geom_uq.py   --mode geom   --checkpoint ./output_new/mast3r_scannetpp_catconv_geom_sanity/checkpoint-step-110000.pth   --dataset "ScanNetpp(split='train', ROOT='/workspace/data/scannetpp_processed_cut3r', resolution=224, aug_crop=16)"   --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(224, 224), head_type='catconv', output_mode='pts3d+desc24', depth_mode=('exp', -float('inf'), float('inf')), conf_mode=('exp', 1, float('inf')), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, float('inf')))"   --output_dir ./vis_s0_geom_110000   --num_samples 16 


S1:
  python tools/vis_scannetpp_geom_uq.py \
    --mode uq3d \
    --checkpoint ./output_new/mast3r_scannetpp_catconv_xyzUQ_freeze/checkpoint-last.pth \
    --dataset "ScanNetpp(split='val', ROOT='/workspace/data/scannetpp_processed_cut3r', resolution=224, aug_crop=16)" \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(224, 224), head_type='catconv', output_mode='pts3d+desc24', depth_mode=('exp', -float('inf'), float('inf')), conf_mode=('exp', 1, float('inf')), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, float('inf')))" \
    --output_dir ./vis_s1_uq3d \
    --num_samples 16
"""

import os
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

import mast3r.utils.path_to_dust3r  # noqa: side-effect: add dust3r to sys.path
from mast3r.model import AsymmetricMASt3R  # noqa

from training import build_dataset  # reuse your train.py helper
from dust3r.utils.device import to_cpu, collate_with_cat


def load_model(model_str, checkpoint, device):
    print(f"[viz] Building model from string:\n{model_str}")
    model = eval(model_str)  # AsymmetricMASt3R(...)
    model.to(device)

    if checkpoint is None or checkpoint == "":
        print("[viz] No checkpoint provided, using random / pretrained weights in the model_str.")
        model.eval()
        return model

    print(f"[viz] Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    # Try common keys: 'model', 'model_state', or raw state dict
    state_dict = None
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            # maybe the dict itself is the state dict
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[viz] Loaded state_dict with missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()
    return model


def _find_gt_depth(view1):
    """Try to find a GT depth map inside view1."""
    for key in ("metric_depth", "depth_gt", "depthmap", "depth"):
        if key in view1 and isinstance(view1[key], torch.Tensor):
            return view1[key]
    return None


def _to_device_view(view, device):
    # view is usually a dict-like object; move all tensors to device
    for k, v in list(view.items()):
        if isinstance(v, torch.Tensor):
            view[k] = v.to(device, non_blocking=True)
    return view


def _imshow_side_by_side(depth_pred, depth_gt, out_path, title_pred="pred", title_gt="gt", cmap="magma", shared=True):
    """depth_* are [H,W] tensors on CPU."""
    depth_pred = depth_pred.detach().cpu()
    if depth_gt is not None:
        depth_gt = depth_gt.detach().cpu()

    def _percentile(x, p):
        x_flat = x.flatten()
        x_flat = x_flat[torch.isfinite(x_flat)]
        if x_flat.numel() == 0:
            return None
        return torch.quantile(x_flat, p / 100.0)

    def _range(x):
        p5 = _percentile(x, 5.0)
        p95 = _percentile(x, 95.0)
        if p5 is None or p95 is None:
            return None, None
        return float(p5), float(p95)

    if shared:
        vmin_p, vmax_p = _range(depth_pred)
        vmin_g, vmax_g = _range(depth_gt) if depth_gt is not None else (None, None)
        vmin = vmin_p if vmin_g is None else min(vmin_p, vmin_g) if vmin_p is not None else vmin_g
        vmax = vmax_p if vmax_g is None else max(vmax_p, vmax_g) if vmax_p is not None else vmax_g
        vmin_pred = vmin_gt = vmin
        vmax_pred = vmax_gt = vmax
    else:
        vmin_pred, vmax_pred = _range(depth_pred)
        vmin_gt, vmax_gt = _range(depth_gt) if depth_gt is not None else (None, None)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_pred, cmap=cmap, vmin=vmin_pred, vmax=vmax_pred)
    plt.title(title_pred)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if depth_gt is not None:
        plt.imshow(depth_gt, cmap=cmap, vmin=vmin_gt, vmax=vmax_gt)
        plt.title(title_gt)
    else:
        plt.imshow(depth_pred, cmap=cmap, vmin=vmin_pred, vmax=vmax_pred)
        plt.title(title_pred)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _imshow_triplet(depth_pred, depth_gt, risk_map, out_path):
    """
    depth_pred, depth_gt, risk_map are [H,W] tensors on CPU.
    Show: predicted depth / error map / risk map.
    """
    depth_pred = depth_pred.detach().cpu()
    if depth_gt is not None:
        depth_gt = depth_gt.detach().cpu()
        err = (depth_pred - depth_gt).abs()
    else:
        depth_gt = torch.zeros_like(depth_pred)
        err = torch.zeros_like(depth_pred)

    risk_map = risk_map.detach().cpu()

    # normalize separately
    def _percentile(x, p):
        x_flat = x.flatten()
        x_flat = x_flat[torch.isfinite(x_flat)]
        if x_flat.numel() == 0:
            return None
        return torch.quantile(x_flat, p / 100.0)

    vmin_d = _percentile(depth_pred, 5.0)
    vmax_d = _percentile(depth_pred, 95.0)
    vmin_e = _percentile(err, 5.0)
    vmax_e = _percentile(err, 95.0)
    vmin_r = _percentile(risk_map, 5.0)
    vmax_r = _percentile(risk_map, 95.0)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(depth_pred, cmap="magma", vmin=vmin_d, vmax=vmax_d)
    plt.title("pred depth")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(err, cmap="viridis", vmin=vmin_e, vmax=vmax_e)
    plt.title("|pred - gt|")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(risk_map, cmap="inferno", vmin=vmin_r, vmax=vmax_r)
    plt.title("risk (xyz_var norm)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def run_s0_geom(args):
    """Sanity check: visualize geometry only (no UQ)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, args.checkpoint, device)

    print(f"[viz] Building val loader: {args.dataset}")
    val_loader = build_dataset(args.dataset, batch_size=1, num_workers=4, test=True)

    os.makedirs(args.output_dir, exist_ok=True)

    saved = 0
    for i, batch in enumerate(val_loader):
        if i < args.offset:
            continue

        if saved >= args.num_samples:
            break

        if isinstance(batch, dict) and "view1" in batch:
            pass
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 2 and isinstance(batch[0], dict):
                batch = {"view1": batch[0], "view2": batch[1]}
            else:
                batch = collate_with_cat(batch)
        else:
            batch = collate_with_cat(batch)

        view1 = batch["view1"]
        view2 = batch["view2"]  # unused, but model expects it

        view1 = _to_device_view(view1, device)
        view2 = _to_device_view(view2, device)

        pred1, pred2 = model(view1, view2)
        # pred1["pts3d"]: [B,3,H,W]
        pts3d_pred = pred1.get("pts3d", None)
        print("[viz][geom] pts3d_pred.shape =", None if pts3d_pred is None else pts3d_pred.shape)
        if "xyz_var" in pred1:
            print("[viz][geom] xyz_var.shape  =", pred1["xyz_var"].shape)
        else:
            print("[viz][geom] xyz_var not in pred1")
        if pts3d_pred is None:
            print("[viz][geom] pred1 has no 'pts3d' key, skipping sample.")
            continue

        # use z-axis as depth proxy; handle both channel-first [B,3,H,W] and channel-last [B,H,W,3]
        pts = pts3d_pred[0]  # drop batch, now [3,H,W] or [H,W,3]

        if pts.ndim == 3 and pts.shape[0] == 3:
            # channel-first: [3, H, W]
            depth_pred = pts[2]  # [H, W]
        elif pts.ndim == 3 and pts.shape[-1] == 3:
            # channel-last: [H, W, 3]
            depth_pred = pts[..., 2]  # [H, W]
        else:
            raise RuntimeError(f"[viz][geom] Unexpected pts3d_pred shape after squeezing batch: {pts.shape}")

        gt_depth = _find_gt_depth(view1)
        if gt_depth is not None:
            gt_depth = gt_depth[0, 0] if gt_depth.ndim == 4 else gt_depth[0]

        print(
            "[S0][geom] depth_pred stats:",
            float(depth_pred.min()),
            float(depth_pred.max()),
        )
        if gt_depth is not None:
            print(
                "[S0][geom] depth_gt   stats:",
                float(gt_depth.min()),
                float(gt_depth.max()),
            )
        else:
            print("[S0][geom] depth_gt   stats: None")

        out_path = os.path.join(args.output_dir, f"s0_geom_{i:06d}_{saved:03d}.png")
        _imshow_side_by_side(
            depth_pred,
            gt_depth,
            out_path,
            title_pred="pred depth (own range)",
            title_gt="GT depth (own range)",
            shared=False,
        )

        saved += 1
        print(f"[viz][geom] Saved {out_path}")


@torch.no_grad()
def run_s1_uq3d(args):
    """3D UQ only: visualize xyz_var risk map vs error."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, args.checkpoint, device)

    print(f"[viz] Building val loader: {args.dataset}")
    val_loader = build_dataset(args.dataset, batch_size=1, num_workers=4, test=True)

    os.makedirs(args.output_dir, exist_ok=True)

    saved = 0
    for i, batch in enumerate(val_loader):
        if i < args.offset:
            continue
        
        if saved >= args.num_samples:
            break

        if isinstance(batch, dict) and "view1" in batch:
            pass
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 2 and isinstance(batch[0], dict):
                batch = {"view1": batch[0], "view2": batch[1]}
            else:
                batch = collate_with_cat(batch)
        else:
            batch = collate_with_cat(batch)

        view1 = batch["view1"]
        view2 = batch["view2"]

        view1 = _to_device_view(view1, device)
        view2 = _to_device_view(view2, device)

        pred1, pred2 = model(view1, view2)

        pts3d_pred = pred1.get("pts3d", None)
        xyz_var = pred1.get("xyz_var", None)
        print("[viz][uq3d] pts3d_pred.shape =", None if pts3d_pred is None else pts3d_pred.shape)
        print("[viz][uq3d] xyz_var.shape    =", None if xyz_var is None else xyz_var.shape)

        if pts3d_pred is None or xyz_var is None:
            print("[viz][uq3d] missing 'pts3d' or 'xyz_var', skipping sample.")
            continue

        # depth_pred from pts3d_pred: handle both [B,3,H,W] and [B,H,W,3]
        pts = pts3d_pred[0]

        if pts.ndim == 3 and pts.shape[0] == 3:
            # [3, H, W]
            depth_pred = pts[2]  # [H, W]
        elif pts.ndim == 3 and pts.shape[-1] == 3:
            # [H, W, 3]
            depth_pred = pts[..., 2]  # [H, W]
        else:
            raise RuntimeError(f"[viz][uq3d] Unexpected pts3d_pred shape after squeezing batch: {pts.shape}")

        # xyz_var risk map: sum variance over x,y,z then sqrt, supports [B,3,H,W] and [B,H,W,3]
        xyz = xyz_var[0]

        if xyz.ndim == 3 and xyz.shape[0] == 3:
            # [3, H, W]
            var_sum = xyz.sum(dim=0)  # [H, W]
        elif xyz.ndim == 3 and xyz.shape[-1] == 3:
            # [H, W, 3]
            var_sum = xyz.sum(dim=-1)  # [H, W]
        else:
            raise RuntimeError(f"[viz][uq3d] Unexpected xyz_var shape after squeezing batch: {xyz.shape}")

        var_norm = torch.sqrt(torch.clamp(var_sum, min=0.0))

        gt_depth = _find_gt_depth(view1)
        if gt_depth is not None:
            gt_depth = gt_depth[0, 0] if gt_depth.ndim == 4 else gt_depth[0]

        out_path = os.path.join(args.output_dir, f"s1_uq3d_{i:06d}_{saved:03d}.png")
        _imshow_triplet(depth_pred, gt_depth, var_norm, out_path)

        saved += 1
        print(f"[viz][uq3d] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["geom", "uq3d"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples in the dataloader")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "geom":
        run_s0_geom(args)
    else:
        run_s1_uq3d(args)


if __name__ == "__main__":
    main()

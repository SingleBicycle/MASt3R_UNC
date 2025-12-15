#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化几何 depth_mu vs feature depth_mean_feat + diff heatmap

用法示例：

  python tools/viz_geom_feat_depth.py \
    --model "AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(224,224), head_type='dpt_uq_feat', output_mode='pts3d', depth_mode=('exp', -float('inf'), float('inf')), conf_mode=('exp', 1, float('inf')), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
    --ckpt ./output/mast3r_scannetpp_uq_feat_phase1_fast_temp1/checkpoint-step-340000.pth \
    --dataset "ScanNetpp(split='val', ROOT='/workspace/data/scannetpp_processed_cut3r', resolution=224, aug_crop=16)" \
    --num_samples 4 \
    --output_dir ./debug_depth_vis

"""


import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

import mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.datasets import get_data_loader  # noqa
from mast3r.model import AsymmetricMASt3R, AsymmetricMASt3RWithDUNEBackbone  # noqa


IGNORE_KEYS = {'depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'}


def _move_to_device(view, device):
    for k in list(view.keys()):
        if k in IGNORE_KEYS:
            continue
        if torch.is_tensor(view[k]):
            view[k] = view[k].to(device, non_blocking=True)
    return view


def _depth_to_numpy(t: torch.Tensor):
    if t is None:
        return None
    if t.ndim == 4:
        # [B,1,H,W]
        t = t[0, 0]
    elif t.ndim == 3:
        # [B,H,W]
        t = t[0]
    return t.detach().cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser(description="Visualize geom vs feat depth for a few samples.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model ctor string, e.g. AsymmetricMASt3R(..., head_type='dpt_uq_feat', ...)")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pth from training.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset spec string, same格式 as training, e.g. ScanNetpp(..., split='test', ...)")
    parser.add_argument("--num_samples", type=int, default=4, help="How many batches to visualize.")
    parser.add_argument("--output_dir", type=str, default="debug_depth_vis", help="Where to save PNGs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[viz] Building dataloader for: {args.dataset}")
    loader = get_data_loader(
        args.dataset,
        batch_size=1,
        num_workers=4,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
    )

    print(f"[viz] Instantiating model: {args.model}")
    model = eval(args.model)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[viz] Loaded checkpoint. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[viz]  missing keys (truncated):", list(missing)[:5])
    if unexpected:
        print("[viz]  unexpected keys (truncated):", list(unexpected)[:5])

    model.to(device)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.num_samples:
                break

            view1, view2 = batch
            view1 = _move_to_device(view1, device)
            view2 = _move_to_device(view2, device)

            pred1, pred2 = model(view1, view2)

            depth_geom = pred1.get("depth_mu", None)
            depth_feat = pred1.get("depth_mean_feat", None)

            if depth_geom is None or depth_feat is None:
                print(f"[viz] Sample {idx}: missing keys depth_mu or depth_mean_feat, skipping.")
                continue

            d_geom = _depth_to_numpy(depth_geom)
            d_feat = _depth_to_numpy(depth_feat)

            if d_geom.shape != d_feat.shape:
                print(f"[viz] Sample {idx}: shape mismatch geom {d_geom.shape} vs feat {d_feat.shape}, skipping.")
                continue

            diff = np.abs(d_geom - d_feat)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            im0 = axes[0].imshow(d_geom)
            axes[0].set_title("geom: depth_mu")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(d_feat)
            axes[1].set_title("feat: depth_mean_feat")
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            im2 = axes[2].imshow(diff)
            axes[2].set_title("|geom - feat|")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()
            out_path = os.path.join(args.output_dir, f"sample_{idx:04d}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[viz] Saved {out_path}")


if __name__ == "__main__":
    main()

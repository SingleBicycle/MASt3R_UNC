import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from mast3r.model import AsymmetricMASt3R
from dust3r.datasets.scannetpp import ScanNetpp  # Use the import path that matches your repo layout
from dust3r.utils.image import rgb  # for de-normalizing images

# Prefer GPU when available, otherwise fall back to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1) Load model + checkpoint
def load_model(ckpt_path):
    model = AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(224, 224),
        head_type="dpt_uq_feat", #for geometric only use "dpt_uq"
        output_mode="pts3d",
        depth_mode=("exp", float("-inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# 2) Grab one ScanNet++ sample
def get_sample(dataset_root, idx=0):
    ds = ScanNetpp(
        split="train",
        ROOT=dataset_root,
        resolution=224,  # Match model training resolution
        aug_crop=0,  # Disable random crop for visualization
    )
    views = ds[idx]  # list of two view dictionaries

    def to_batched(view):
        batched = {}
        for k, v in view.items():
            if k == "true_shape":
                batched[k] = torch.as_tensor(v, device=device).unsqueeze(0)
            elif torch.is_tensor(v):
                batched[k] = v.unsqueeze(0).to(device)
            elif isinstance(v, np.ndarray):
                batched[k] = torch.from_numpy(v).unsqueeze(0).to(device)
            else:
                batched[k] = v  # metadata / strings / ints
        return batched

    return to_batched(views[0]), to_batched(views[1])

def viz_one_pair(model, sample, out_dir="viz_uq_`00000_phase2", idx=0):
    view1, view2 = sample
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        pred1, pred2 = model(view1, view2)

    print("pred1 keys:", pred1.keys())

    # Try to pull UQ-related outputs from the prediction dict
    # Names depend on your implementation: use depth_mu/depth_var if present,
    # otherwise compute mean/variance from gamma/nu/alpha/beta
    if "depth_mu_feat" in pred1 and "depth_var_feat" in pred1:
        depth_mu = pred1["depth_mu_feat"][0, 0].detach().cpu().numpy()
        depth_var = pred1["depth_var_feat"][0, 0].detach().cpu().numpy()
    elif "depth_mu" in pred1 and "depth_var" in pred1:
        depth_mu = pred1["depth_mu"][0, 0].detach().cpu().numpy()
        depth_var = pred1["depth_var"][0, 0].detach().cpu().numpy()
    elif "depth_gamma_feat" in pred1:
        gamma = pred1["depth_gamma_feat"][0, 0].detach().cpu().numpy()
        nu    = pred1["depth_nu_feat"][0, 0].detach().cpu().numpy()
        alpha = pred1["depth_alpha_feat"][0, 0].detach().cpu().numpy()
        beta  = pred1["depth_beta_feat"][0, 0].detach().cpu().numpy()
        depth_mu = gamma
        depth_var = beta * (1.0 + nu) / (nu * (alpha - 1.0) + 1e-6)
    elif "depth_gamma" in pred1:
        gamma = pred1["depth_gamma"][0, 0].detach().cpu().numpy()
        nu    = pred1["depth_nu"][0, 0].detach().cpu().numpy()
        alpha = pred1["depth_alpha"][0, 0].detach().cpu().numpy()
        beta  = pred1["depth_beta"][0, 0].detach().cpu().numpy()
        depth_mu = gamma
        depth_var = beta * (1.0 + nu) / (nu * (alpha - 1.0) + 1e-6)
    else:
        raise RuntimeError("Cannot find depth UQ keys in pred1")

    # De-normalize for visualization
    rgb_img = rgb(view1["img"][0])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input RGB")
    plt.imshow(rgb_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Depth mean (mu)")
    plt.imshow(depth_mu, cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Depth variance (UQ)")
    # Apply log to variance for a clearer visualization
    v_show = np.log(depth_var + 1e-6)
    plt.imshow(v_show, cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis("off")

    out_path = out_dir / f"sample{idx}_uq_with_feat_100000.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("saved to", out_path)

if __name__ == "__main__":
    ckpt_path = "./output/mast3r_scannetpp_uq_feat_phase2_fast/checkpoint-step-100000.pth"  
    dataset_root = "/workspace/data/scannetpp_processed_cut3r"

    model = load_model(ckpt_path)
    # sample random indices to visualize different pairs
    ds_len = len(ScanNetpp(split="train", ROOT=dataset_root, resolution=224, aug_crop=0))
    rng = np.random.default_rng()
    sample = get_sample(dataset_root, idx=0)
    viz_one_pair(model, sample, idx=0)
    # for i in range(10):
    #     idx = int(rng.integers(ds_len))
    #     sample = get_sample(dataset_root, idx=idx)
    #     viz_one_pair(model, sample, idx=idx)

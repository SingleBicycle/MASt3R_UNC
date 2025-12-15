import torch, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mast3r.model import AsymmetricMASt3R
from dust3r.datasets.scannetpp import ScanNetpp
from dust3r.utils.image import rgb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(ckpt):
    model = AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(224, 224),
        head_type="dpt_uq_feat",
        output_mode="pts3d",
        depth_mode=("exp", float("-inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
        dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    return model.to(device).eval()

def get_sample(root, idx=0):
    ds = ScanNetpp(split="train", ROOT=root, resolution=224, aug_crop=0)
    v1, v2 = ds[idx]
    def to_batched(v):
        out = {}
        for k, val in v.items():
            if k == "true_shape":
                out[k] = torch.as_tensor(val, device=device).unsqueeze(0)
            elif torch.is_tensor(val):
                out[k] = val.unsqueeze(0).to(device)
            elif isinstance(val, np.ndarray):
                out[k] = torch.from_numpy(val).unsqueeze(0).to(device)
            else:
                out[k] = val
        return out
    return to_batched(v1), to_batched(v2)

def viz_one(model, sample, out_dir, idx):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    v1, v2 = sample
    with torch.no_grad():
        pred1, _ = model(v1, v2)

    print("pred1 keys:", pred1.keys())

    # Phase 1: force deterministic feature depth; evidential head is frozen
    depth_mean_feat = pred1.get("depth_mean_feat")
    if depth_mean_feat is None:
        raise RuntimeError(f"depth_mean_feat not found in prediction dict: {pred1.keys()}")
    depth_mean_feat = depth_mean_feat[0, 0].detach().cpu().numpy()

    # Optional geometry teacher for comparison/diff (if present in checkpoint)
    depth_geom = pred1.get("depth_mu")
    depth_geom = depth_geom[0, 0].detach().cpu().numpy() if depth_geom is not None else None

    rgb_img = rgb(v1["img"][0])

    print(f"feat mean range: {depth_mean_feat.min():.4f} .. {depth_mean_feat.max():.4f}")
    if "depth_mu_feat" in pred1:
        dm = pred1["depth_mu_feat"][0, 0].detach().cpu().numpy()
        print(f"mu_feat(gamma) range (frozen head): {dm.min():.4f} .. {dm.max():.4f}")
    if depth_geom is not None:
        print(f"geom depth_mu range: {depth_geom.min():.4f} .. {depth_geom.max():.4f}")

    ncols = 3 if depth_geom is not None else 2
    plt.figure(figsize=(12 if depth_geom is not None else 8, 4))
    plt.subplot(1, ncols, 1); plt.title("Input RGB"); plt.imshow(rgb_img); plt.axis("off")
    plt.subplot(1, ncols, 2); plt.title("Depth mean (feat, Phase1)"); plt.imshow(depth_mean_feat, cmap="viridis")
    plt.colorbar(fraction=0.046, pad=0.04); plt.axis("off")

    if depth_geom is not None:
        diff = depth_mean_feat - depth_geom
        plt.subplot(1, ncols, 3)
        plt.title("Feat - Geom depth")
        im = plt.imshow(diff, cmap="bwr")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis("off")
        print(f"feat-geom diff range: {diff.min():.4f} .. {diff.max():.4f}")

    out_path = out_dir / f"phase1_feat_depth_{idx}.png"
    plt.tight_layout(); plt.savefig(out_path, dpi=150)
    print("saved to", out_path)

if __name__ == "__main__":
    ckpt =  "./output/mast3r_scannetpp_uq_feat_phase1_fast_temp/checkpoint-step-100000.pth" #"./output/mast3r_scannetpp_uq_feat_phase1_fast/checkpoint-step-80000.pth"  # 改成你的
    data_root = "/workspace/data/scannetpp_processed_cut3r"                      # 改成你的
    model = load_model(ckpt)
    sample = get_sample(data_root, idx=0)
    viz_one(model, sample, out_dir="viz_uq_phase1_100000_distillation1.0", idx=23)

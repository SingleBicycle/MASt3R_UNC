import torch, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mast3r.model import AsymmetricMASt3R
from dust3r.datasets.scannetpp import ScanNetpp
from dust3r.utils.image import rgb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- 1) 加载几何 UQ 模型（Option A） ----------
def load_geom_uq_model(ckpt_path):
    model = AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(224, 224),
        head_type="dpt_uq",          # 几何 UQ 版本
        output_mode="pts3d",
        depth_mode=("exp", float("-inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
        dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd), strict=False)
    return model.to(device).eval()

# ---------- 2) 取一个 ScanNet++ 样本 ----------
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

# ---------- 3) 画 2×2 图 ----------
def viz_fig1_geom_uq(model, sample, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    v1, v2 = sample
    with torch.no_grad():
        pred1, _ = model(v1, v2)

    print("pred1 keys:", pred1.keys())

    # 1) RGB
    rgb_img = rgb(v1["img"][0])

    # 2) baseline depth：优先用 depth_mu，没有就从 pts3d 取 z
    if "depth_mu" in pred1:
        depth_base = pred1["depth_mu"][0, 0].detach().cpu().numpy()
    else:
        pts3d = pred1["pts3d"][0].detach().cpu().numpy()  # [3,H,W]
        depth_base = pts3d[2]                             # z 轴当 depth

    # 做个简单的可视化裁剪，避免远处极端值拉伸 colormap
    d_min, d_max = np.percentile(depth_base, [5, 95])
    depth_vis = np.clip(depth_base, d_min, d_max)

    # 3) 原始 conf（MASt3R 自带的）
    conf = pred1.get("conf", None)
    if conf is None:
        raise RuntimeError("pred1 里没有 'conf'，看看 keys 是什么再改脚本")
    # conf can be (B,H,W) or (B,1,H,W); keep spatial dims
    conf = conf[0] if conf.ndim == 3 else conf[0, 0]
    conf = conf.detach().cpu().numpy()

    # 4) Geo UQ variance（我们 evidential 学出来的）
    depth_var = pred1.get("depth_var", None)
    if depth_var is None:
        raise RuntimeError("pred1 里没有 'depth_var'，确认你用的是 dpt_uq checkpoint")
    # depth_var can be (B,1,H,W) or (B,H,W)
    depth_var = depth_var[0, 0] if depth_var.ndim == 4 else depth_var[0]
    depth_var = depth_var.detach().cpu().numpy()
    # 为了看得清楚，用 log
    depth_var_vis = np.log(depth_var + 1e-6)

    print(f"depth_base range: {depth_base.min():.3f} .. {depth_base.max():.3f}")
    print(f"conf range      : {conf.min():.3f} .. {conf.max():.3f}")
    print(f"depth_var range : {depth_var.min():.3f} .. {depth_var.max():.3f}")

    plt.figure(figsize=(10, 8))

    # 左上：RGB
    plt.subplot(2, 2, 1)
    plt.title("Input RGB")
    plt.imshow(rgb_img)
    plt.axis("off")

    # 右上：baseline depth
    plt.subplot(2, 2, 2)
    plt.title("Baseline depth (geom)")
    im = plt.imshow(depth_vis, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")

    # 左下：原始 conf
    plt.subplot(2, 2, 3)
    plt.title("Original confidence (MASt3R)")
    im = plt.imshow(conf, cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")

    # 右下：Geo UQ variance (log)
    plt.subplot(2, 2, 4)
    plt.title("Geo UQ: log(depth variance)")
    im = plt.imshow(depth_var_vis, cmap="magma")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("saved fig1 to", out_path)


if __name__ == "__main__":
    ckpt_geom_uq = "./output/mast3r_scannetpp_uq_geom_headonly/checkpoint-step-850000.pth"  # 改成你的
    data_root = "/workspace/data/scannetpp_processed_cut3r"                                 # 改成你的

    model_geom = load_geom_uq_model(ckpt_geom_uq)
    sample = get_sample(data_root, idx=0)  # 可以多试几个 idx
    viz_fig1_geom_uq(model_geom, sample, out_path="viz_icml/fig1_geom_vs_conf.png")

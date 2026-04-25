#!/usr/bin/env python3
"""Extract SAE latents from images and visualize activated neurons sorted by magnitude.

Usage:
    python src/sae_rae/visualize_latents.py \
        --images /path/to/images/*.jpg \
        --ckpt ckpts/ft_sae_cls_e10_ffhq/ep-0010.pt \
        --out results/sae_latents
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sae_rae.script_utils import add_sys_path, full_torch_load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="+", required=True, help="Input image paths")
    p.add_argument("--ckpt", required=True, help="SAE-RAE checkpoint (ep-XXXX.pt) — config embedded")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--out", type=str, default="results/sae_latents")
    p.add_argument("--top-k", type=int, default=0, help="Show only top-k neurons (0 = all active)")
    return p.parse_args()


def load_conditioner(ckpt_path: str, device: torch.device):
    """Load DINOv2 + SAE conditioner using config embedded in checkpoint.
    Falls back to strict=False for SAE loading to handle version mismatches.
    """
    ckpt = full_torch_load(ckpt_path, map_location="cpu")
    full_cfg = ckpt["config"]

    sae_cond_cfg = full_cfg.get("sae_condition")
    rae_cfg = full_cfg.get("stage_1")
    s1_params = rae_cfg.get("params", {})
    project_root = Path(__file__).resolve().parent.parent

    # Try strict=True first; fall back to strict=False for old checkpoints
    from sae_rae.conditioning import DinoClsSaeConditioner
    try:
        conditioner = DinoClsSaeConditioner(
            encoder_config_path=str(sae_cond_cfg.get("encoder_config_path",
                                                       s1_params.get("encoder_config_path"))),
            dinov2_path=str(sae_cond_cfg.get("dinov2_path",
                                              s1_params.get("encoder_params", {}).get("dinov2_path"))),
            encoder_input_size=int(sae_cond_cfg.get("encoder_input_size",
                                                      s1_params.get("encoder_input_size", 224))),
            sae_ckpt_path=str(sae_cond_cfg.get("sae_ckpt")),
            sae_src_path=str(sae_cond_cfg.get("sae_src_path", str(project_root / "src"))),
        ).to(device)
    except RuntimeError:
        # Old checkpoint (e.g. BatchTopK without threshold_ema) — load SAE with strict=False
        print("[warn] strict SAE load failed; retrying with strict=False (version mismatch)", flush=True)
        conditioner = _load_conditioner_nonstrict(sae_cond_cfg, s1_params, device)

    conditioner.eval().requires_grad_(False)
    return conditioner


def _load_conditioner_nonstrict(sae_cond_cfg, s1_params, device: torch.device):
    """Build conditioner by loading SAE with strict=False."""
    import torch.nn as nn
    from transformers import AutoImageProcessor, Dinov2WithRegistersModel

    encoder_config_path = str(sae_cond_cfg.get("encoder_config_path",
                                                  s1_params.get("encoder_config_path")))
    dinov2_path = str(sae_cond_cfg.get("dinov2_path",
                                        s1_params.get("encoder_params", {}).get("dinov2_path")))
    encoder_input_size = int(sae_cond_cfg.get("encoder_input_size",
                                               s1_params.get("encoder_input_size", 224)))
    sae_ckpt_path = str(sae_cond_cfg.get("sae_ckpt"))
    project_root = Path(__file__).resolve().parent.parent
    sae_src_path = str(sae_cond_cfg.get("sae_src_path", str(project_root / "src")))

    import sys
    if sae_src_path not in sys.path:
        sys.path.insert(0, sae_src_path)
    from sae_local.model import Autoencoder  # type: ignore

    sae_raw = full_torch_load(sae_ckpt_path, map_location="cpu")
    sae_state = sae_raw.get("sae_state_dict", sae_raw)
    sae = Autoencoder.from_state_dict(sae_state, strict=False)
    sae.requires_grad_(False)

    proc = AutoImageProcessor.from_pretrained(encoder_config_path)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std  = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    dino = Dinov2WithRegistersModel.from_pretrained(dinov2_path)
    dino.requires_grad_(False)

    class _Conditioner(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            self.encoder_input_size = encoder_input_size
            self.dino = dino
            self.sae = sae

        @property
        def cond_dim(self):
            return int(self.sae.latent_bias.numel())

        @torch.no_grad()
        def forward(self, images):
            _, _, h, w = images.shape
            if h != self.encoder_input_size or w != self.encoder_input_size:
                images = nn.functional.interpolate(
                    images,
                    size=(self.encoder_input_size, self.encoder_input_size),
                    mode="bicubic", align_corners=False,
                )
            x = (images - self.mean.to(images.device)) / self.std.to(images.device)
            cls = self.dino(x).last_hidden_state[:, 0, :]
            latents, _ = self.sae.encode(cls)
            latents = nn.functional.normalize(latents, p=2, dim=-1)
            return latents

    return _Conditioner().to(device)


def plot_latent(ax, latent: np.ndarray, title: str, top_k: int = 0):
    """Bar chart of activated neurons sorted by magnitude (descending)."""
    nonzero_mask = latent != 0
    active_vals = latent[nonzero_mask]
    active_idx = np.where(nonzero_mask)[0]

    if len(active_vals) == 0:
        ax.text(0.5, 0.5, "No active neurons", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return

    # Sort by magnitude descending
    order = np.argsort(-np.abs(active_vals))
    sorted_vals = active_vals[order]
    sorted_idx = active_idx[order]

    if top_k > 0:
        sorted_vals = sorted_vals[:top_k]
        sorted_idx = sorted_idx[:top_k]

    n = len(sorted_vals)
    colors = ["#4682C8" if v >= 0 else "#C85A46" for v in sorted_vals]
    x_pos = np.arange(n)

    ax.bar(x_pos, sorted_vals, color=colors, width=0.8, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.5)

    # X-tick labels: neuron index
    if n <= 40:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in sorted_idx], fontsize=6, rotation=90)
    else:
        # Show every Nth label
        step = max(1, n // 30)
        ax.set_xticks(x_pos[::step])
        ax.set_xticklabels([str(i) for i in sorted_idx[::step]], fontsize=6, rotation=90)

    sparsity = 100.0 * (1 - nonzero_mask.sum() / len(latent))
    subtitle = f"active={nonzero_mask.sum()}/{len(latent)}  sparsity={sparsity:.1f}%"
    ax.set_title(f"{title}\n{subtitle}", fontsize=8)
    ax.set_ylabel("activation", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    add_sys_path(project_root / "vendor" / "rae_src")
    add_sys_path(project_root / "src")

    from utils.train_utils import center_crop_arr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[info] Loading conditioner ...", flush=True)
    conditioner = load_conditioner(args.ckpt, device)
    cond_dim = conditioner.cond_dim
    print(f"[info] SAE latent dim: {cond_dim}", flush=True)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.ToTensor(),
    ])

    # Collect and sort image paths
    img_paths = sorted(Path(p) for p in args.images)
    n = len(img_paths)
    print(f"[info] Processing {n} images ...", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    latents = []
    names = []

    for path in img_paths:
        with Image.open(path) as pil:
            pil = pil.convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(device)

        with torch.no_grad():
            cond = conditioner(tensor)[0]  # (cond_dim,)

        lat = cond.float().cpu().numpy()
        latents.append(lat)
        names.append(path.stem)

        nonzero = (lat != 0).sum()
        print(f"  {path.name}  active={nonzero}/{cond_dim}  "
              f"sparsity={100*(cond_dim-nonzero)/cond_dim:.1f}%", flush=True)

    # -----------------------------------------------------------------------
    # 1. Individual charts per image  (image | bar chart side by side)
    # -----------------------------------------------------------------------
    for path, name, lat in zip(img_paths, names, latents):
        fig = plt.figure(figsize=(16, 3.8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5], wspace=0.08)

        # left: original image
        ax_img = fig.add_subplot(gs[0])
        pil_disp = Image.open(path).convert("RGB")
        ax_img.imshow(pil_disp)
        ax_img.axis("off")
        ax_img.set_title(name, fontsize=9)

        # right: bar chart
        ax_bar = fig.add_subplot(gs[1])
        plot_latent(ax_bar, lat, title="", top_k=args.top_k)

        fig.tight_layout()
        out_path = out_dir / f"{name}_sae_latent.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [saved] {out_path}", flush=True)

    # -----------------------------------------------------------------------
    # 2. Combined grid figure (all images, each row = image + bar chart)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 4.0 * n))
    outer = gridspec.GridSpec(n, 1, hspace=0.55)

    for i, (path, name, lat) in enumerate(zip(img_paths, names, latents)):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i],
                                                 width_ratios=[1, 6], wspace=0.07)
        ax_img = fig.add_subplot(inner[0])
        pil_disp = Image.open(path).convert("RGB")
        ax_img.imshow(pil_disp)
        ax_img.axis("off")
        ax_img.set_title(name, fontsize=9)

        ax_bar = fig.add_subplot(inner[1])
        plot_latent(ax_bar, lat, title="", top_k=args.top_k)

    fig.suptitle("SAE Latent Activations (sorted by magnitude)", fontsize=13, y=1.002)
    combined_path = out_dir / "all_sae_latents.png"
    fig.savefig(combined_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] combined figure → {combined_path}", flush=True)

    # -----------------------------------------------------------------------
    # 3. Heatmap: images × neurons (top-K most active across all images)
    # -----------------------------------------------------------------------
    mat = np.stack(latents, axis=0)  # (N, cond_dim)
    max_per_neuron = np.abs(mat).max(axis=0)
    top_neurons = np.argsort(-max_per_neuron)[:80]  # top-80 neurons across dataset
    sub_mat = mat[:, top_neurons]  # (N, 80)

    fig, ax = plt.subplots(figsize=(20, 0.45 * n + 1.5))
    vmax = np.abs(sub_mat).max() or 1.0
    im = ax.imshow(sub_mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(range(len(top_neurons)))
    ax.set_xticklabels([str(i) for i in top_neurons], fontsize=6, rotation=90)
    ax.set_xlabel("Neuron index (top-80 by max activation)", fontsize=9)
    ax.set_title("SAE Latent Heatmap — Images × Neurons", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    heatmap_path = out_dir / "heatmap_sae_latents.png"
    fig.savefig(heatmap_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] heatmap → {heatmap_path}", flush=True)


if __name__ == "__main__":
    main()

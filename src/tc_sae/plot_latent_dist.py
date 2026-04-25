#!/usr/bin/env python3
"""SAE latent activation distribution visualization.

각 뉴런의 pre-activation 혹은 non-zero post-activation 분포를 시각화하고
Gaussian / Laplace 분포 fit을 비교합니다.

Usage:
    python src/tc_sae/plot_latent_dist.py \\
        --ckpt src/checkpoints/ffhq256_topk_tc_preact_v9/sae_tc_latest.pt \\
        --features /scratch/x3411a10/unconditional_diffusion/SAE-DINO/features/ffhq256/eval_features.bin \\
        --out plots/latent_dist \\
        [--mode pre_act|post_act_nonzero] \\
        [--n-plot 32] \\
        [--max-samples 20000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sae_local.model import Autoencoder


def load_sae(ckpt_path: str, device: torch.device) -> Autoencoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["sae_state_dict"]
    sae = Autoencoder.from_state_dict(sd, strict=False)
    sae = sae.to(device)
    sae.eval()
    return sae


def collect_activations(
    sae: Autoencoder,
    features_path: str,
    device: torch.device,
    batch_size: int,
    max_samples: int,
    mode: str,
) -> np.ndarray:
    """
    Returns array of shape [N, n_latents].
    mode='pre_act'           : raw encoder output before activation (continuous)
    mode='post_act_nonzero'  : post-TopK values (zeros replaced with NaN for per-neuron analysis)
    """
    p = Path(features_path)
    shape_path = p.parent / (p.stem + "_shape.npy")
    shape = tuple(int(v) for v in np.load(str(shape_path)))
    feats = np.memmap(str(p), dtype=np.float32, mode="r", shape=shape)

    n_total = min(len(feats), max_samples)
    n_latents = sae.encoder.weight.shape[0]

    # pre-allocate output
    out = np.empty((n_total, n_latents), dtype=np.float32)

    idx = 0
    with torch.no_grad():
        while idx < n_total:
            end = min(idx + batch_size, n_total)
            x = torch.from_numpy(feats[idx:end].copy()).to(device)
            x_norm, info = sae.preprocess(x)
            z_pre = sae.encode_pre_act(x_norm)  # [B, n_latents]

            if mode == "pre_act":
                out[idx:end] = z_pre.float().cpu().numpy()
            else:  # post_act_nonzero
                latents = sae.activation(z_pre)  # [B, n_latents], sparse
                vals = latents.float().cpu().numpy()
                vals[vals == 0.0] = np.nan  # mask zeros so per-neuron stats skip them
                out[idx:end] = vals

            idx = end
            if idx % 10000 == 0 or idx == n_total:
                print(f"  collected {idx}/{n_total}", flush=True)

    return out  # [N, n_latents]


def fit_and_plot_neuron(
    ax: plt.Axes,
    data: np.ndarray,
    neuron_idx: int,
    mode: str,
) -> dict:
    """Plot histogram of one neuron's activation values + Gaussian/Laplace fits.
    Returns dict with fit stats."""
    if mode == "post_act_nonzero":
        vals = data[:, neuron_idx]
        vals = vals[~np.isnan(vals)]  # keep only non-zero activations
    else:
        vals = data[:, neuron_idx]

    if len(vals) < 10:
        ax.set_title(f"#{neuron_idx}\n(dead)", fontsize=7)
        return {"neuron": neuron_idx, "dead": True}

    # fit Gaussian
    mu_g, std_g = stats.norm.fit(vals)
    # fit Laplace
    mu_l, b_l = stats.laplace.fit(vals)

    # histogram
    n_bins = min(60, max(20, len(vals) // 50))
    counts, bin_edges, _ = ax.hist(vals, bins=n_bins, density=True,
                                   color="#4C72B0", alpha=0.55, label="data")

    x = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu_g, std_g),
            "r-", lw=1.5, label=f"Gauss σ={std_g:.2f}")
    ax.plot(x, stats.laplace.pdf(x, mu_l, b_l),
            "g--", lw=1.5, label=f"Laplace b={b_l:.2f}")

    # KS test (lower D = better fit)
    ks_gauss = stats.kstest(vals, "norm", args=(mu_g, std_g)).statistic
    ks_laplace = stats.kstest(vals, "laplace", args=(mu_l, b_l)).statistic
    kurt = float(stats.kurtosis(vals))  # excess kurtosis; Gauss=0, Laplace=3

    better = "Laplace" if ks_laplace < ks_gauss else "Gauss"
    ax.set_title(
        f"#{neuron_idx}  n={len(vals)}\n"
        f"KS: G={ks_gauss:.3f} L={ks_laplace:.3f}  kurt={kurt:.2f}\n"
        f"→ {better}",
        fontsize=6.5,
    )
    ax.tick_params(labelsize=5)
    ax.legend(fontsize=4.5, loc="upper right")

    return {
        "neuron": neuron_idx,
        "dead": False,
        "n_samples": len(vals),
        "mu": float(mu_g),
        "std": float(std_g),
        "ks_gauss": float(ks_gauss),
        "ks_laplace": float(ks_laplace),
        "kurtosis": kurt,
    }


def plot_summary(stats_list: list[dict], out_dir: Path, mode: str) -> None:
    """Summary plot: KS statistics and kurtosis distribution across all neurons."""
    alive = [s for s in stats_list if not s.get("dead", False)]
    if not alive:
        return

    ks_g = np.array([s["ks_gauss"] for s in alive])
    ks_l = np.array([s["ks_laplace"] for s in alive])
    kurt = np.array([s["kurtosis"] for s in alive])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Latent distribution fit summary  ({mode})", fontsize=12)

    # KS comparison per neuron
    ax = axes[0]
    x = np.arange(len(alive))
    ax.scatter(x, ks_g, s=2, alpha=0.4, label="KS Gauss", color="red")
    ax.scatter(x, ks_l, s=2, alpha=0.4, label="KS Laplace", color="green")
    ax.axhline(ks_g.mean(), color="red", lw=1, ls="--", label=f"mean G={ks_g.mean():.3f}")
    ax.axhline(ks_l.mean(), color="green", lw=1, ls="--", label=f"mean L={ks_l.mean():.3f}")
    ax.set_xlabel("neuron (sorted by index)")
    ax.set_ylabel("KS statistic (lower = better fit)")
    ax.legend(fontsize=8)
    ax.set_title("KS distance per neuron")

    # histogram of KS difference (Gauss - Laplace): positive → Laplace better
    ax = axes[1]
    diff = ks_g - ks_l
    frac_laplace = (diff > 0).mean()
    ax.hist(diff, bins=50, color="#4C72B0", alpha=0.7)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("KS_Gauss - KS_Laplace\n(>0 means Laplace fits better)")
    ax.set_ylabel("count")
    ax.set_title(f"Laplace better: {100*frac_laplace:.1f}% of neurons")

    # kurtosis distribution (Gauss=0, Laplace=3)
    ax = axes[2]
    ax.hist(kurt, bins=50, color="#55A868", alpha=0.7)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="Gauss (κ=0)")
    ax.axvline(3, color="green", lw=1.5, ls="--", label="Laplace (κ=3)")
    ax.set_xlabel("Excess kurtosis")
    ax.set_ylabel("count")
    ax.set_title(f"Kurtosis  mean={kurt.mean():.2f}  median={np.median(kurt):.2f}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f"summary_{mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="SAE checkpoint .pt")
    parser.add_argument("--features", type=str, required=True, help="cached features .bin")
    parser.add_argument("--out", type=str, default="plots/latent_dist")
    parser.add_argument("--mode", type=str, default="pre_act",
                        choices=["pre_act", "post_act_nonzero"],
                        help="pre_act: raw encoder output; post_act_nonzero: non-zero TopK values")
    parser.add_argument("--n-plot", type=int, default=32,
                        help="number of sample neurons to plot in the grid")
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="max data samples to use")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neuron-ids", type=str, default=None,
                        help="comma-separated specific neuron IDs to plot (overrides --n-plot random sample)")
    parser.add_argument("--all-stats", action="store_true",
                        help="compute KS stats for ALL neurons (slow but gives full summary)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading SAE from {args.ckpt}")
    sae = load_sae(args.ckpt, device)
    n_latents = sae.encoder.weight.shape[0]
    print(f"[info] SAE n_latents={n_latents}")

    print(f"[info] collecting activations (mode={args.mode}, max_samples={args.max_samples})")
    data = collect_activations(sae, args.features, device,
                               args.batch_size, args.max_samples, args.mode)
    print(f"[info] data shape: {data.shape}")

    # pick neurons to plot
    if args.neuron_ids is not None:
        plot_ids = [int(x) for x in args.neuron_ids.split(",")]
    else:
        plot_ids = np.random.choice(n_latents, size=min(args.n_plot, n_latents), replace=False).tolist()
        plot_ids = sorted(plot_ids)

    # grid plot
    n_cols = 8
    n_rows = (len(plot_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.5))
    axes_flat = axes.flatten() if n_rows * n_cols > 1 else [axes]

    stats_sample = []
    for ax_i, (ax, nid) in enumerate(zip(axes_flat, plot_ids)):
        s = fit_and_plot_neuron(ax, data, nid, args.mode)
        stats_sample.append(s)

    # hide unused axes
    for ax in axes_flat[len(plot_ids):]:
        ax.set_visible(False)

    label = "pre-activation" if args.mode == "pre_act" else "non-zero post-activation"
    fig.suptitle(f"SAE latent {label} distributions\n{Path(args.ckpt).name}  n_latents={n_latents}",
                 fontsize=10)
    plt.tight_layout()
    grid_path = out_dir / f"grid_{args.mode}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {grid_path}")

    # full summary stats (optional: all neurons)
    if args.all_stats:
        print("[info] computing KS stats for all neurons...")
        all_stats = []
        for nid in range(n_latents):
            if args.mode == "post_act_nonzero":
                vals = data[:, nid]
                vals = vals[~np.isnan(vals)]
            else:
                vals = data[:, nid]

            if len(vals) < 10:
                all_stats.append({"neuron": nid, "dead": True})
                continue

            mu_g, std_g = stats.norm.fit(vals)
            mu_l, b_l = stats.laplace.fit(vals)
            ks_g = stats.kstest(vals, "norm", args=(mu_g, std_g)).statistic
            ks_l = stats.kstest(vals, "laplace", args=(mu_l, b_l)).statistic
            all_stats.append({
                "neuron": nid,
                "dead": False,
                "ks_gauss": float(ks_g),
                "ks_laplace": float(ks_l),
                "kurtosis": float(stats.kurtosis(vals)),
            })
            if (nid + 1) % 500 == 0:
                print(f"  {nid+1}/{n_latents}", flush=True)

        plot_summary(all_stats, out_dir, args.mode)

        # save csv
        import csv
        csv_path = out_dir / f"stats_{args.mode}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["neuron", "dead", "ks_gauss", "ks_laplace", "kurtosis"])
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"[saved] {csv_path}")

        alive = [s for s in all_stats if not s.get("dead", False)]
        if alive:
            ks_g = np.array([s["ks_gauss"] for s in alive])
            ks_l = np.array([s["ks_laplace"] for s in alive])
            kurt = np.array([s["kurtosis"] for s in alive])
            print(f"\n=== Summary ({len(alive)} alive / {n_latents} total) ===")
            print(f"  KS Gauss:   mean={ks_g.mean():.4f}  median={np.median(ks_g):.4f}")
            print(f"  KS Laplace: mean={ks_l.mean():.4f}  median={np.median(ks_l):.4f}")
            print(f"  Kurtosis:   mean={kurt.mean():.3f}  median={np.median(kurt):.3f}  (Gauss=0, Laplace=3)")
            print(f"  Laplace fits better: {100*(ks_g > ks_l).mean():.1f}% of neurons")
    else:
        plot_summary(stats_sample, out_dir, args.mode)


if __name__ == "__main__":
    main()

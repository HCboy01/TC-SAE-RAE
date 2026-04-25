#!/usr/bin/env python3
"""독립 Gaussian 샘플링으로 SAE latent를 생성하고 RAE로 이미지 디코딩.

각 뉴런의 pre-activation 분포에서 독립적으로 샘플 → TopK → flow model → 이미지.

Usage:
    python src/sae_rae/sample_gaussian.py \
        --config configs/sae_rae/ImageNet256/DiTDH-XL_DINOv2-B_SAECLS_TC_v8best.yaml \
        --rae-ckpt /scratch/x3411a10/unconditional_diffusion/SAE-RAE/results/ffhq256_TC_v8best_ep40/best.pt \
        --features /scratch/.../eval_features.bin \
        --n-samples 16 \
        --out plots/gaussian_samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import amp
from torchvision.utils import save_image, make_grid

SRC_ROOT = Path(__file__).resolve().parent.parent   # .../TC-SAE-RAE/src
PROJECT_ROOT = SRC_ROOT.parent                      # .../TC-SAE-RAE
for _p in [str(SRC_ROOT), str(PROJECT_ROOT / "vendor" / "rae_src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sae_rae.script_utils import add_sys_path, full_torch_load, safe_torch_load


def compute_neuron_stats(sae, features_path: str, device, batch_size: int = 512,
                         max_samples: int = 20000) -> tuple[torch.Tensor, torch.Tensor]:
    """각 뉴런의 pre-activation μ와 σ를 eval 데이터에서 계산."""
    p = Path(features_path)
    shape = tuple(int(v) for v in np.load(str(p.parent / (p.stem + "_shape.npy"))))
    feats = np.memmap(str(p), dtype=np.float32, mode="r", shape=shape)
    n = min(len(feats), max_samples)

    sum1 = torch.zeros(sae.encoder.weight.shape[0], device=device)
    sum2 = torch.zeros_like(sum1)
    count = 0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(feats[i:i+batch_size].copy()).to(device)
            x_norm, _ = sae.preprocess(x)
            z_pre = sae.encode_pre_act(x_norm)
            sum1 += z_pre.sum(0)
            sum2 += z_pre.pow(2).sum(0)
            count += z_pre.shape[0]

    mu = sum1 / count
    var = sum2 / count - mu.pow(2)
    sigma = var.clamp(min=1e-6).sqrt()
    print(f"[stats] μ: mean={mu.mean():.4f} std={mu.std():.4f}  "
          f"σ: mean={sigma.mean():.4f} std={sigma.std():.4f}")
    return mu, sigma


def sample_cond(mu: torch.Tensor, sigma: torch.Tensor, k: int, n: int,
                device) -> torch.Tensor:
    """μ, σ에서 독립 Gaussian 샘플 → TopK 적용 → sparse cond."""
    n_latents = mu.shape[0]
    # [n, n_latents] 샘플
    z_pre = mu.unsqueeze(0) + sigma.unsqueeze(0) * torch.randn(n, n_latents, device=device)
    # TopK
    topk = torch.topk(z_pre, k=k, dim=-1)
    import torch.nn.functional as F
    values = F.relu(topk.values)
    cond = torch.zeros_like(z_pre)
    cond.scatter_(-1, topk.indices, values)
    active = (cond > 0).float().sum(dim=1).mean().item()
    print(f"[sample] sampled {n} conds  avg_active={active:.1f}/{n_latents}")
    return cond


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--rae-ckpt", type=str, required=True,
                        help="RAE flow model checkpoint (best.pt or ep-XXXX.pt)")
    parser.add_argument("--features", type=str, required=True,
                        help="cached eval features .bin for computing per-neuron stats")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--out", type=str, default="plots/gaussian_samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-stat-samples", type=int, default=20000)
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from stage1 import RAE
    from stage2.transport import create_transport, Sampler
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs
    from sae_local.model import Autoencoder

    full_cfg = OmegaConf.load(args.config)
    (rae_config, model_config, transport_config, _, _, _, training_config, _) = parse_configs(full_cfg)
    training_cfg = dict(OmegaConf.to_container(training_config, resolve=True)) if training_config else {}
    misc_cfg = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True))
    sae_cond_cfg = full_cfg.get("sae_condition")

    null_label = int(training_cfg.get("null_label", misc_cfg.get("num_classes", 1000)))
    latent_size = tuple(int(v) for v in misc_cfg.get("latent_size", [768, 16, 16]))

    # ── SAE 로드 (통계 계산용) ─────────────────────────────────────────────
    print("[info] loading SAE ...", flush=True)
    sae_ckpt_path = str(sae_cond_cfg.get("sae_ckpt"))
    sae_ckpt = torch.load(sae_ckpt_path, map_location="cpu", weights_only=False)
    sae_state = sae_ckpt.get("sae_state_dict", sae_ckpt)
    sae = Autoencoder.from_state_dict(dict(sae_state), strict=False).to(device)
    sae.eval().requires_grad_(False)

    n_latents = sae.encoder.weight.shape[0]
    k = sae.activation.k if hasattr(sae.activation, "k") else 64
    print(f"[info] SAE n_latents={n_latents}  k={k}")

    # ── 뉴런별 통계 계산 ──────────────────────────────────────────────────
    print("[info] computing per-neuron stats ...", flush=True)
    mu, sigma = compute_neuron_stats(sae, args.features, device,
                                     max_samples=args.max_stat_samples)

    # ── RAE 로드 ──────────────────────────────────────────────────────────
    print("[info] loading RAE ...", flush=True)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval().requires_grad_(False)

    # ── flow model 로드 ───────────────────────────────────────────────────
    print("[info] loading flow model ...", flush=True)
    if "params" not in model_config:
        model_config["params"] = {}
    model_config["params"]["cond_dim"] = n_latents

    stage2_ckpt = model_config.get("ckpt", None)
    if stage2_ckpt is not None:
        model_config = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
        model_config.pop("ckpt", None)

    model = instantiate_from_config(model_config).to(device)
    if stage2_ckpt is not None:
        base_state = safe_torch_load(stage2_ckpt, map_location="cpu")
        base_state = base_state.get("ema", base_state.get("model", base_state))
        model.load_state_dict(base_state, strict=False)

    ckpt = full_torch_load(args.rae_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["ema"], strict=False)
    model.eval().requires_grad_(False)

    # ── transport ─────────────────────────────────────────────────────────
    transport_params = dict(transport_config.get("params", {}))
    shift_dim = int(misc_cfg.get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(misc_cfg.get("time_dist_shift_base", 4096))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params,
                                 time_dist_shift=(shift_dim / shift_base) ** 0.5)
    sample_fn = Sampler(transport).sample_ode(
        sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3)

    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    N = args.n_samples
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 독립 Gaussian 샘플 생성 ───────────────────────────────────────────
    print("[info] sampling from independent Gaussians ...", flush=True)
    cond_gauss = sample_cond(mu, sigma, k, N, device)

    print("[info] generating images ...", flush=True)

    generated_list = []
    for i in range(N):
        c = cond_gauss[i:i+1]
        y_null = torch.full((1,), null_label, device=device, dtype=torch.long)

        with torch.no_grad():
            if args.cfg_scale > 1.0:
                z0 = torch.randn(1, *latent_size, device=device)
                z = torch.cat([z0, z0], dim=0)
                y_cfg = torch.cat([y_null, y_null], dim=0)
                cond_cfg = torch.cat([c, torch.zeros_like(c)], dim=0)

                def model_fn(x, t, y, cfg_scale, cfg_interval=(0.0, 1.0), cond=None):
                    out = model.forward(x, t, y, cond=cond)
                    eps, rest = out[:, :model.in_channels], out[:, model.in_channels:]
                    c_eps, u_eps = eps[:1], eps[1:]
                    guided = u_eps + cfg_scale * (c_eps - u_eps)
                    return torch.cat([torch.cat([guided, guided], dim=0), rest], dim=1)

                with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    zhat = sample_fn(z, model_fn, y=y_cfg, cond=cond_cfg,
                                     cfg_scale=args.cfg_scale,
                                     cfg_interval=(0.0, 1.0))[-1][:1]
                    img = rae.decode(zhat).clamp(0, 1)
            else:
                z0 = torch.randn(1, *latent_size, device=device)
                with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    zhat = sample_fn(z0, model.forward, y=y_null, cond=c)[-1]
                    img = rae.decode(zhat).clamp(0, 1)

        generated_list.append(img.cpu())
        print(f"  [{i+1}/{N}] done", flush=True)
        torch.cuda.empty_cache()

    generated = torch.cat(generated_list, dim=0)

    # ── 저장 ──────────────────────────────────────────────────────────────
    grid = make_grid(generated.cpu(), nrow=4, padding=2)
    grid_path = out_dir / "gaussian_samples.png"
    save_image(grid, grid_path)
    print(f"[saved] {grid_path}")

    # 개별 이미지도 저장
    for i, img in enumerate(generated):
        save_image(img, out_dir / f"sample_{i:03d}.png")
    print(f"[saved] {N} individual images → {out_dir}/")


if __name__ == "__main__":
    main()

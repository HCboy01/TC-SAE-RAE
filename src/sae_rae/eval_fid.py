#!/usr/bin/env python3
"""Standalone FID evaluation for two checkpoints using the same 1000 conditioning images.

Usage:
    python src/sae_rae/eval_fid.py \
        --ckpt1 ckpts/ft_sae_cls_ffhq_topk_l2norm_ep10_condgate_rerun/sae_cond_ft/ep-0006.pt \
        --config1 configs/sae_rae/ImageNet256/DiTDH-XL_DINOv2-B_SAECLS_ep100_l2norm.yaml \
        --ckpt2 ckpts/ft_sae_cls_batch_topk_ep50/ep-0018.pt \
        --config2 configs/sae_rae/ImageNet256/DiTDH-XL_DINOv2-B_SAECLS_ep100.yaml \
        --cond-path /home/gimhyeongchan97/datasets/ffhq256/imagefolder/val \
        --real-path /home/gimhyeongchan97/datasets/ffhq256/imagefolder/val \
        --num-samples 1000 \
        --cfg-scale 2.0 \
        --out results/fid_compare
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import amp
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sae_rae.script_utils import (
    RecursiveImageDataset,
    add_sys_path,
    full_torch_load,
    safe_torch_load,
)


# ---------------------------------------------------------------------------
def load_model_bundle(ckpt_path: str, config_path: str, device: torch.device):
    """Load RAE + conditioner + DiT for a given checkpoint and config."""
    from stage1 import RAE
    from stage2.transport import create_transport, Sampler
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import center_crop_arr, parse_configs
    from sae_rae.conditioning import DinoClsSaeConditioner

    full_cfg = OmegaConf.load(config_path)
    (rae_config, model_config, transport_config, _, _, _, training_config, _) = parse_configs(full_cfg)

    misc_cfg = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True))
    latent_size = tuple(int(v) for v in misc_cfg.get("latent_size", [768, 16, 16]))
    training_cfg = dict(OmegaConf.to_container(training_config, resolve=True)) if training_config else {}
    null_label = int(training_cfg.get("null_label", misc_cfg.get("num_classes", 1000)))

    sae_cond_cfg = full_cfg.get("sae_condition")
    s1_params = rae_config.get("params", {})

    # RAE
    print(f"  [load] RAE ...", flush=True)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval().requires_grad_(False)

    # Conditioner
    print(f"  [load] DINOv2 + SAE conditioner ...", flush=True)
    conditioner = DinoClsSaeConditioner(
        encoder_config_path=str(sae_cond_cfg.get("encoder_config_path", s1_params.get("encoder_config_path"))),
        dinov2_path=str(sae_cond_cfg.get("dinov2_path", s1_params.get("encoder_params", {}).get("dinov2_path"))),
        encoder_input_size=int(sae_cond_cfg.get("encoder_input_size", s1_params.get("encoder_input_size", 224))),
        sae_ckpt_path=str(sae_cond_cfg.get("sae_ckpt")),
        sae_src_path=str(project_root / "src") if sae_cond_cfg.get("sae_src_path") is None else str(sae_cond_cfg.get("sae_src_path")),
    ).to(device)
    conditioner.eval().requires_grad_(False)

    # DiT
    print(f"  [load] DiT model ...", flush=True)
    if "params" not in model_config:
        model_config["params"] = {}
    model_config["params"]["cond_dim"] = conditioner.cond_dim

    stage2_ckpt = model_config.get("ckpt", None)
    if stage2_ckpt is not None:
        model_config = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
        model_config.pop("ckpt", None)

    model = instantiate_from_config(model_config).to(device)
    if stage2_ckpt is not None:
        base_state = safe_torch_load(stage2_ckpt, map_location="cpu")
        base_state = base_state.get("ema", base_state.get("model", base_state))
        model.load_state_dict(base_state, strict=False)

    ckpt = full_torch_load(ckpt_path, map_location="cpu")
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        raise RuntimeError(f"Checkpoint {ckpt_path} has neither 'ema' nor 'model'.")
    model.eval().requires_grad_(False)

    # Transport / sampler
    transport_params = dict(transport_config.get("params", {}))
    shift_dim = int(full_cfg.get("misc", {}).get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(full_cfg.get("misc", {}).get("time_dist_shift_base", 4096))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=(shift_dim / shift_base) ** 0.5)
    sample_fn = Sampler(transport).sample_ode(sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3)

    return rae, conditioner, model, sample_fn, latent_size, null_label


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    rae, conditioner, model, sample_fn,
    latent_size, null_label,
    loader, num_samples, out_dir,
    cfg_scale, cfg_t_min, cfg_t_max,
    precision, device,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    generated = 0
    img_idx = 0

    print(f"  Generating {num_samples} samples → {out_dir}", flush=True)

    for images, _ in loader:
        if generated >= num_samples:
            break
        images = images.to(device, non_blocking=True)
        bsz = images.size(0)
        remain = num_samples - generated
        if bsz > remain:
            images = images[:remain]
            bsz = remain

        z0 = torch.randn(bsz, *latent_size, device=device)
        cond = conditioner(images)

        if cfg_scale > 1.0:
            z = torch.cat([z0, z0], dim=0)

            def model_fn(x, t, cfg_scale, cfg_interval=(0.0, 1.0), cond=None):
                half = x[: len(x) // 2]
                combined = torch.cat([half, half], dim=0)
                cond_half = cond[: cond.shape[0] // 2] if cond.shape[0] == x.shape[0] else cond
                cond_combined = torch.cat([cond_half, torch.zeros_like(cond_half)], dim=0)
                model_out = model.forward(combined, t, cond=cond_combined)
                eps, rest = model_out[:, : model.in_channels], model_out[:, model.in_channels :]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                t_half = t[: len(t) // 2]
                guided_eps = torch.where(
                    ((t_half >= cfg_interval[0]) & (t_half <= cfg_interval[1])).view(-1, *[1] * (len(cond_eps.shape) - 1)),
                    uncond_eps + cfg_scale * (cond_eps - uncond_eps),
                    cond_eps,
                )
                return torch.cat([torch.cat([guided_eps, guided_eps], dim=0), rest], dim=1)

            model_kwargs = dict(cond=cond, cfg_scale=cfg_scale, cfg_interval=(cfg_t_min, cfg_t_max))
        else:
            z = z0
            model_fn = model.forward
            model_kwargs = dict(cond=cond)

        with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            zhat = sample_fn(z, model_fn, **model_kwargs)[-1]
            if cfg_scale > 1.0:
                zhat = zhat[:bsz]
            samples = rae.decode(zhat).clamp(0, 1)

        for i in range(samples.size(0)):
            save_image(samples[i], out_dir / f"{img_idx:06d}.png", normalize=False)
            img_idx += 1

        generated += bsz
        print(f"  [{generated}/{num_samples}]", flush=True)

    print(f"  Done. {img_idx} images saved.", flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt1", required=True)
    p.add_argument("--config1", required=True)
    p.add_argument("--ckpt2", required=True)
    p.add_argument("--config2", required=True)
    p.add_argument("--cond-path", required=True, help="Directory with conditioning images")
    p.add_argument("--real-path", required=True, help="Directory with real images for FID")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--cfg-t-min", type=float, default=0.0)
    p.add_argument("--cfg-t-max", type=float, default=1.0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=str, default="results/fid_compare")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent.parent
    add_sys_path(project_root / "vendor" / "rae_src")
    add_sys_path(project_root / "src")

    from utils.train_utils import center_crop_arr

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    # ---- shared conditioning dataset (sorted, first N images) ----
    cond_transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.ToTensor(),
    ])
    cond_dataset = RecursiveImageDataset(
        args.cond_path,
        transform=cond_transform,
        max_samples=args.num_samples,
        return_path=True,
    )
    cond_loader = DataLoader(
        cond_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Conditioning dataset: {len(cond_dataset)} images from {args.cond_path}", flush=True)

    results = {}

    for tag, ckpt_path, config_path in [
        ("model1_condgate_ep6", args.ckpt1, args.config1),
        ("model2_batchtopk_ep18", args.ckpt2, args.config2),
    ]:
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating: {tag}", flush=True)
        print(f"  ckpt  : {ckpt_path}", flush=True)
        print(f"  config: {config_path}", flush=True)

        rae, conditioner, model, sample_fn, latent_size, null_label = load_model_bundle(
            ckpt_path, config_path, device
        )

        sample_dir = out_base / tag / "samples"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)

        generate_samples(
            rae, conditioner, model, sample_fn,
            latent_size, null_label,
            cond_loader, args.num_samples, sample_dir,
            args.cfg_scale, args.cfg_t_min, args.cfg_t_max,
            args.precision, device,
        )

        print(f"  Computing FID vs {args.real_path} ...", flush=True)
        metrics = calculate_metrics(
            input1=str(sample_dir),
            input2=args.real_path,
            fid=True,
            cuda=(device.type == "cuda"),
            batch_size=64,
            samples_find_deep=True,
            isc=False,
            kid=False,
            prc=False,
        )
        fid = float(metrics["frechet_inception_distance"])
        results[tag] = fid
        print(f"  FID ({tag}): {fid:.4f}", flush=True)

        # free GPU memory before loading next model
        del rae, conditioner, model, sample_fn
        torch.cuda.empty_cache()

    print(f"\n{'='*60}", flush=True)
    print("=== FID Results ===", flush=True)
    for tag, fid in results.items():
        print(f"  {tag}: {fid:.4f}", flush=True)

    result_file = out_base / "fid_results.txt"
    with open(result_file, "w") as f:
        f.write(f"num_samples: {args.num_samples}\n")
        f.write(f"cond_path:   {args.cond_path}\n")
        f.write(f"real_path:   {args.real_path}\n")
        f.write(f"cfg_scale:   {args.cfg_scale}\n\n")
        for tag, fid in results.items():
            f.write(f"{tag}: {fid:.4f}\n")
    print(f"\nResults saved to {result_file}", flush=True)


if __name__ == "__main__":
    main()

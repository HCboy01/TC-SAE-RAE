#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import amp
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sae_rae.script_utils import (
    RecursiveImageDataset,
    add_sys_path,
    full_torch_load,
    safe_torch_load,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CFG grids from SAE-conditioned samples")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True, help="epoch checkpoint (ep-XXXX.pt)")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cond-path", type=str, default="", help="If empty, use fid_eval.cond_path from config")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--scales", type=str, default="1.5,2.0,3.0")
    p.add_argument("--triplet-only", action="store_true", help="Save only 3-column triplet grids (input/no-CFG/CFG)")
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    project_root = Path(__file__).resolve().parent.parent
    add_sys_path(project_root / "vendor" / "rae_src")
    add_sys_path(project_root / "src")

    from stage1 import RAE
    from stage2.transport import create_transport, Sampler
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import center_crop_arr, parse_configs
    from sae_rae.conditioning import DinoClsSaeConditioner

    full_cfg = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        _sampler_config,
        _guidance_config,
        _misc,
        training_config,
        _eval_config,
    ) = parse_configs(full_cfg)

    training_cfg = OmegaConf.to_container(training_config, resolve=True) if training_config is not None else {}
    training_cfg = dict(training_cfg)
    fid_cfg = dict(OmegaConf.to_container(full_cfg.get("fid_eval", {}), resolve=True)) if full_cfg.get("fid_eval", None) is not None else {}
    misc_cfg = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True)) if full_cfg.get("misc", None) is not None else {}

    num_classes = int(misc_cfg.get("num_classes", 1000))
    null_label = int(training_cfg.get("null_label", num_classes))

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval().requires_grad_(False)

    sae_cond_cfg = full_cfg.get("sae_condition", None)
    if sae_cond_cfg is None:
        raise ValueError("Config must include top-level `sae_condition` block.")

    s1_params = rae_config.get("params", {})
    encoder_config_path = str(sae_cond_cfg.get("encoder_config_path", s1_params.get("encoder_config_path")))
    encoder_input_size = int(sae_cond_cfg.get("encoder_input_size", s1_params.get("encoder_input_size", 224)))
    encoder_params = s1_params.get("encoder_params", {})
    dinov2_path = str(sae_cond_cfg.get("dinov2_path", encoder_params.get("dinov2_path", encoder_config_path)))
    sae_ckpt_path = str(sae_cond_cfg.get("sae_ckpt"))
    sae_src_path = str(sae_cond_cfg.get("sae_src_path", str(project_root / "src")))

    conditioner = DinoClsSaeConditioner(
        encoder_config_path=encoder_config_path,
        dinov2_path=dinov2_path,
        encoder_input_size=encoder_input_size,
        sae_ckpt_path=sae_ckpt_path,
        sae_src_path=sae_src_path,
    ).to(device)
    conditioner.eval().requires_grad_(False)

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
        if "ema" in base_state:
            base_state = base_state["ema"]
        elif "model" in base_state:
            base_state = base_state["model"]
        model.load_state_dict(base_state, strict=False)

    ckpt = full_torch_load(args.ckpt, map_location="cpu")
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"], strict=False)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        raise RuntimeError("Checkpoint has neither 'ema' nor 'model'.")
    model.eval().requires_grad_(False)

    transport_params = dict(transport_config.get("params", {}))
    shift_dim = int(full_cfg.get("misc", {}).get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(full_cfg.get("misc", {}).get("time_dist_shift_base", 4096))
    time_dist_shift = (shift_dim / shift_base) ** 0.5
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3)

    cond_path = args.cond_path or str(fid_cfg.get("cond_path", ""))
    if not cond_path:
        raise ValueError("cond_path is empty; pass --cond-path or set fid_eval.cond_path")

    cond_transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = RecursiveImageDataset(cond_path, transform=cond_transform, return_path=True)
    loader = DataLoader(dataset, batch_size=args.num_pairs, shuffle=False, num_workers=2, pin_memory=True)
    images_cond, _ = next(iter(loader))
    images_cond = images_cond.to(device, non_blocking=True)

    latent_size = tuple(int(v) for v in misc_cfg.get("latent_size", [768, 16, 16]))
    bsz = images_cond.shape[0]
    y_null = torch.full((bsz,), null_label, device=device, dtype=torch.long)
    cond = conditioner(images_cond)

    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def model_fn_cond_cfg(x, t, y, cfg_scale, cfg_interval=(0.0, 1.0), cond=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        y_half = y[: len(y) // 2]
        y_combined = torch.cat([y_half, y_half], dim=0)

        if cond is None:
            raise ValueError("cond is required")
        if cond.shape[0] == x.shape[0]:
            cond_half = cond[: cond.shape[0] // 2]
        elif cond.shape[0] == x.shape[0] // 2:
            cond_half = cond
        else:
            raise ValueError(f"Invalid cond batch size: cond={cond.shape[0]}, x={x.shape[0]}")

        cond_uncond = torch.zeros_like(cond_half)
        cond_combined = torch.cat([cond_half, cond_uncond], dim=0)

        model_out = model.forward(combined, t, y_combined, cond=cond_combined)
        eps, rest = model_out[:, : model.in_channels], model_out[:, model.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        t_half = t[: len(t) // 2]
        guid_t_min, guid_t_max = cfg_interval
        guided_eps = torch.where(
            ((t_half >= guid_t_min) & (t_half <= guid_t_max)).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps),
            cond_eps,
        )
        eps = torch.cat([guided_eps, guided_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    for i, scale in enumerate(scales, start=1):
        z0 = torch.randn(bsz, *latent_size, device=device)
        z = torch.cat([z0, z0], dim=0)
        y = torch.cat([y_null, y_null], dim=0)
        model_kwargs = dict(y=y, cond=cond, cfg_scale=scale, cfg_interval=(0.0, 1.0))

        with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            zhat_nocfg = sample_fn(z0, model.forward, y=y_null, cond=cond)[-1]
            samples_nocfg = rae.decode(zhat_nocfg).clamp(0, 1)

            zhat = sample_fn(z, model_fn_cond_cfg, **model_kwargs)[-1]
            zhat = zhat[:bsz]
            samples_cfg = rae.decode(zhat).clamp(0, 1)

        pair_tensors = []
        inputs_vis = images_cond.detach().cpu()
        outputs_vis = samples_cfg.detach().cpu()
        outputs_nocfg_vis = samples_nocfg.detach().cpu()
        if not args.triplet_only:
            for j in range(bsz):
                pair_tensors.append(inputs_vis[j])
                pair_tensors.append(outputs_vis[j])
            grid = make_grid(torch.stack(pair_tensors), nrow=2, normalize=False)

            out_path = out_dir / f"cfg_grid_{i:02d}_scale_{scale:.2f}.png"
            save_image(grid, out_path, normalize=False)
            print(f"saved: {out_path}", flush=True)

        triplet_tensors = []
        for j in range(bsz):
            triplet_tensors.append(inputs_vis[j])
            triplet_tensors.append(outputs_nocfg_vis[j])
            triplet_tensors.append(outputs_vis[j])
        triplet_grid = make_grid(torch.stack(triplet_tensors), nrow=3, normalize=False)
        triplet_out_path = out_dir / f"triplet_grid_{i:02d}_scale_{scale:.2f}.png"
        save_image(triplet_grid, triplet_out_path, normalize=False)
        print(f"saved: {triplet_out_path}", flush=True)


if __name__ == "__main__":
    main()

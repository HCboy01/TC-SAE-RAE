#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image

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
    parser = argparse.ArgumentParser(description="Generate input/sample pair grids from an SAE-RAE checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cond-path", default="")
    parser.add_argument(
        "--reference-grid-dir",
        default="",
        help="Optional pair_grid directory to crop the left-column input images from.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-pairs", type=int, default=32)
    parser.add_argument("--pairs-per-grid", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--cfg-t-min", type=float, default=0.0)
    parser.add_argument("--cfg-t-max", type=float, default=1.0)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


class ReferenceGridInputDataset(Dataset):
    def __init__(self, grid_dir: str, image_size: int, pairs_per_grid: int):
        self.grid_paths = sorted(Path(grid_dir).glob("pair_grid_*.png"))
        if not self.grid_paths:
            raise FileNotFoundError(f"No pair_grid_*.png files found in {grid_dir}")
        self.image_size = int(image_size)
        self.pairs_per_grid = int(pairs_per_grid)
        self.items = []
        step = self.image_size + 2
        for grid_path in self.grid_paths:
            with Image.open(grid_path) as img:
                width, height = img.size
            max_rows = max((height - 2) // step, 0)
            rows = min(self.pairs_per_grid, max_rows)
            for row in range(rows):
                box = (
                    2,
                    2 + row * step,
                    2 + self.image_size,
                    2 + row * step + self.image_size,
                )
                self.items.append((grid_path, box, f"{grid_path.name}:row{row}"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        grid_path, box, label = self.items[idx]
        with Image.open(grid_path) as img:
            crop = img.convert("RGB").crop(box)
        tensor = transforms.functional.to_tensor(crop)
        return tensor, label


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
    from stage2.transport import Sampler, create_transport
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import center_crop_arr, parse_configs
    from sae_rae.conditioning import DinoClsSaeConditioner
    full_cfg = OmegaConf.load(args.config)
    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        _guidance_config,
        _misc,
        _training_config,
        _eval_config,
    ) = parse_configs(full_cfg)

    fid_cfg = dict(OmegaConf.to_container(full_cfg.get("fid_eval", {}), resolve=True)) if full_cfg.get("fid_eval", None) is not None else {}
    misc_cfg = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True)) if full_cfg.get("misc", None) is not None else {}

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval().requires_grad_(False)

    sae_cond_cfg = full_cfg.get("sae_condition", None)
    if sae_cond_cfg is None:
        raise ValueError("Config must include sae_condition.")
    s1_params = rae_config.get("params", {})
    encoder_params = s1_params.get("encoder_params", {})
    conditioner = DinoClsSaeConditioner(
        encoder_config_path=str(sae_cond_cfg.get("encoder_config_path", s1_params.get("encoder_config_path"))),
        dinov2_path=str(sae_cond_cfg.get("dinov2_path", encoder_params.get("dinov2_path"))),
        encoder_input_size=int(sae_cond_cfg.get("encoder_input_size", s1_params.get("encoder_input_size", 224))),
        sae_ckpt_path=str(sae_cond_cfg.get("sae_ckpt")),
        sae_src_path=str(sae_cond_cfg.get("sae_src_path", str(project_root / "src"))),
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
        raise RuntimeError(f"Checkpoint has neither ema nor model: {args.ckpt}")
    model.eval().requires_grad_(False)

    transport_params = dict(transport_config.get("params", {}))
    shift_dim = int(misc_cfg.get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(misc_cfg.get("time_dist_shift_base", 4096))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=(shift_dim / shift_base) ** 0.5)

    sampler_cfg = dict(OmegaConf.to_container(sampler_config, resolve=True)) if sampler_config is not None else {
        "mode": "ODE",
        "params": {"sampling_method": "dopri5", "num_steps": 50, "atol": 1e-6, "rtol": 1e-3},
    }
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(**dict(sampler_cfg.get("params", {})))

    if args.reference_grid_dir:
        dataset = ReferenceGridInputDataset(args.reference_grid_dir, args.image_size, args.pairs_per_grid)
    else:
        cond_path = args.cond_path or str(fid_cfg.get("cond_path", ""))
        if not cond_path:
            raise ValueError("Pass --cond-path, set fid_eval.cond_path, or pass --reference-grid-dir.")

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ]
        )
        dataset = RecursiveImageDataset(cond_path, transform=transform, return_path=True)
    loader = DataLoader(
        dataset,
        batch_size=args.pairs_per_grid,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    latent_size = tuple(int(v) for v in misc_cfg.get("latent_size", [768, 16, 16]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    saved = 0
    grid_idx = 0
    with torch.no_grad():
        for images_cond, _paths in loader:
            if saved >= args.num_pairs:
                break
            n_pairs = min(images_cond.size(0), args.num_pairs - saved)
            images_cond = images_cond[:n_pairs].to(device, non_blocking=True)
            cond = conditioner(images_cond).float()
            z0 = torch.randn(n_pairs, *latent_size, device=device, dtype=torch.float32)

            if args.cfg_scale > 1.0:
                z = torch.cat([z0, z0], dim=0)
                model_fn = model.forward_with_cfg
                model_kwargs = dict(y=None, cond=cond, cfg_scale=args.cfg_scale, cfg_interval=(args.cfg_t_min, args.cfg_t_max))
            else:
                z = z0
                model_fn = model.forward
                model_kwargs = dict(cond=cond)

            with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                zhat = sample_fn(z, model_fn, **model_kwargs)[-1]
                if args.cfg_scale > 1.0:
                    zhat = zhat[:n_pairs]
                samples = rae.decode(zhat).clamp(0, 1)

            pair_tensors = []
            for i in range(n_pairs):
                pair_tensors.append(images_cond[i].detach().cpu())
                pair_tensors.append(samples[i].detach().cpu())
            grid = make_grid(torch.stack(pair_tensors), nrow=2, normalize=False)
            out_path = out_dir / f"pair_grid_{grid_idx:04d}.png"
            save_image(grid, out_path, normalize=False)
            print(f"saved: {out_path}", flush=True)
            grid_idx += 1
            saved += n_pairs

    print(f"done: saved {saved} pairs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

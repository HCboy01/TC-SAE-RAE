#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import shutil
import json
import re
import sys
from pathlib import Path
from time import time
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import amp

try:
    import wandb as _wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch_fidelity import calculate_metrics
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


def _collect_epoch_checkpoints(out_dir: Path) -> list[tuple[int, Path]]:
    ckpts: list[tuple[int, Path]] = []
    pattern = re.compile(r"^ep-(\d{4})\.pt$")
    for p in out_dir.glob("ep-*.pt"):
        m = pattern.match(p.name)
        if m is not None:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def _prune_old_checkpoints(out_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    ckpts = _collect_epoch_checkpoints(out_dir)
    if len(ckpts) <= keep_last_n:
        return
    to_delete = ckpts[:-keep_last_n]
    for _, p in to_delete:
        try:
            p.unlink()
            print(f"[info] pruned old checkpoint: {p}", flush=True)
        except Exception as e:
            print(f"[warn] failed to prune checkpoint {p}: {e}", flush=True)


class CenterCropTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image):
        import numpy as np
        s = self.image_size
        while min(*pil_image.size) >= 2 * s:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )
        scale = s / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
        arr = np.array(pil_image)
        cy = (arr.shape[0] - s) // 2
        cx = (arr.shape[1] - s) // 2
        return Image.fromarray(arr[cy: cy + s, cx: cx + s])


class OfflineAugTransform:
    def __init__(self, image_size: int, hflip: bool):
        self.hflip = bool(hflip)
        self.base = transforms.Compose(
            [
                CenterCropTransform(image_size),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, pil_image):
        if self.hflip:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.base(pil_image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RAE DiT with SAE-CLS conditioning (AdaLN-only).")
    parser.add_argument("--config", type=str, required=True, help="YAML config")
    parser.add_argument("--data-path", type=str, required=True, help="ImageFolder path")
    parser.add_argument(
        "--cached-latents-dir",
        type=str,
        default="",
        help="Optional directory produced by cache_latents.py. Supports z+cond caches or z-only caches that recompute cond on the fly.",
    )
    parser.add_argument("--results-dir", type=str, default="ckpts/s2_sae_cond", help="Output directory")
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume checkpoint path or 'latest' to auto-pick newest ep-*.pt in output directory.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming, do not restore optimizer state even if present in checkpoint.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="SAE-RAE")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1.0 - decay)


UnlabeledImageDataset = RecursiveImageDataset


class CachedLatentDataset(Dataset):
    """Memmap-backed dataset of precomputed RAE latents and optional SAE conditions."""

    def __init__(self, root: str):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Cached latent path does not exist: {self.root}")

        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing cache metadata: {metadata_path}")
        with metadata_path.open("r") as f:
            self.metadata = json.load(f)

        self.z = np.load(self.root / "z.npy", mmap_mode="r")
        cond_path = self.root / "cond.npy"
        self.cond = np.load(cond_path, mmap_mode="r") if cond_path.exists() else None
        self.has_cond = self.cond is not None
        self.image_size = int(self.metadata.get("image_size", 256))

        self.path_records = None
        if self.has_cond:
            if len(self.z) != len(self.cond):
                raise ValueError(f"Cache length mismatch: z={len(self.z)} cond={len(self.cond)}")
        else:
            paths_path = self.root / "paths.jsonl"
            if not paths_path.exists():
                raise FileNotFoundError(
                    f"Missing cond.npy and paths.jsonl in cache dir: {self.root}. "
                    "Need cond.npy for fully cached training or paths.jsonl to recompute cond from images."
                )
            self.path_records = []
            with paths_path.open("r") as f:
                for line in f:
                    self.path_records.append(json.loads(line))
            if len(self.z) != len(self.path_records):
                raise ValueError(f"Cache length mismatch: z={len(self.z)} paths={len(self.path_records)}")

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        z = torch.from_numpy(np.array(self.z[idx], copy=True))
        if self.has_cond:
            cond = torch.from_numpy(np.array(self.cond[idx], copy=True))
            return {"z": z, "cond": cond}

        record = self.path_records[idx]
        image_path = Path(record["path"])
        if not image_path.exists():
            raise FileNotFoundError(f"Source image missing for cache sample {idx}: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = OfflineAugTransform(self.image_size, hflip=bool(record.get("hflip", False)))(image)
        return {"z": z, "image": image}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script currently expects CUDA.")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    rae_src = project_root / "vendor" / "rae_src"
    sae_rae_src = project_root / "src"

    add_sys_path(rae_src)
    add_sys_path(sae_rae_src)

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
    if rae_config is None or model_config is None or transport_config is None:
        raise ValueError("Config must contain stage_1, stage_2 and transport sections.")

    training_cfg = OmegaConf.to_container(training_config, resolve=True) if training_config is not None else {}
    training_cfg = dict(training_cfg)
    misc_cfg = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True)) if full_cfg.get("misc", None) is not None else {}
    fid_cfg = dict(OmegaConf.to_container(full_cfg.get("fid_eval", {}), resolve=True)) if full_cfg.get("fid_eval", None) is not None else {}
    val_cfg = dict(OmegaConf.to_container(full_cfg.get("val_eval", {}), resolve=True)) if full_cfg.get("val_eval", None) is not None else {}

    num_classes = int(misc_cfg.get("num_classes", 1000))

    # ----- wandb (init before model load to avoid fork OOM) -----
    exp_name = os.environ.get("EXPERIMENT_NAME", "sae_cond_ft")
    wandb_run = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed. Run: pip install wandb")
        wandb_run = _wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or exp_name,
            config=OmegaConf.to_container(full_cfg, resolve=True),
            resume="allow",
        )

    # ----- models -----
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    rae.requires_grad_(False)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    conditioner.eval()
    conditioner.requires_grad_(False)

    if "params" not in model_config:
        model_config["params"] = {}
    model_config["params"]["cond_dim"] = conditioner.cond_dim

    # Load base DiT checkpoint with strict=False because SAE condition adapters
    # introduce extra parameters (cond_*).
    stage2_ckpt = model_config.get("ckpt", None)
    resume_requested = bool(args.resume.strip())
    if stage2_ckpt is not None:
        model_config = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
        model_config.pop("ckpt", None)

    model = instantiate_from_config(model_config).to(device)
    if stage2_ckpt is not None and not resume_requested:
        state_dict = safe_torch_load(stage2_ckpt, map_location="cpu")
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        keys = model.load_state_dict(state_dict, strict=False)
        if len(keys.missing_keys) > 0:
            print(f"[info] stage2 missing keys (expected for cond adapters): {keys.missing_keys}")
        if len(keys.unexpected_keys) > 0:
            print(f"[warn] unexpected stage2 keys: {keys.unexpected_keys}")
        del state_dict
        gc.collect()
    elif stage2_ckpt is not None:
        print("[info] --resume set; skipping base stage2 checkpoint load", flush=True)
    model.train()

    # freeze all except AdaLN + condition adapters
    for name, p in model.named_parameters():
        trainable = ("adaLN_modulation" in name) or ("cond_" in name)
        p.requires_grad_(trainable)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check model names for AdaLN/cond adapters.")

    ema_model = instantiate_from_config(model_config).to(device)
    ema_model.load_state_dict(model.state_dict(), strict=True)
    ema_model.eval()
    ema_model.requires_grad_(False)

    lr = float(training_cfg.get("optimizer", {}).get("lr", 1e-4))
    wd = float(training_cfg.get("optimizer", {}).get("weight_decay", 0.0))
    betas = training_cfg.get("optimizer", {}).get("betas", [0.9, 0.95])
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=wd, betas=tuple(betas))

    transport_params = dict(transport_config.get("params", {}))
    shift_dim = int(full_cfg.get("misc", {}).get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(full_cfg.get("misc", {}).get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)

    use_cached_latents = bool(args.cached_latents_dir.strip())
    if use_cached_latents:
        dataset = CachedLatentDataset(args.cached_latents_dir)
        cache_meta = dataset.metadata
        if dataset.has_cond:
            cached_cond_shape = cache_meta.get("cond_shape")
            cached_cond_dim = int(cached_cond_shape[-1]) if cached_cond_shape is not None else int(dataset.cond.shape[-1])
            if cached_cond_dim != conditioner.cond_dim:
                raise ValueError(
                    f"Cached cond_dim={cached_cond_dim} does not match conditioner cond_dim={conditioner.cond_dim}"
                )
            print(
                f"[info] using cached z+cond: {args.cached_latents_dir} "
                f"num_samples={len(dataset)} z_shape={tuple(dataset.z.shape[1:])} cond_shape={tuple(dataset.cond.shape[1:])}",
                flush=True,
            )
        else:
            print(
                f"[info] using cached z only: {args.cached_latents_dir} "
                f"num_samples={len(dataset)} z_shape={tuple(dataset.z.shape[1:])} cond=recomputed_from_images",
                flush=True,
            )
    else:
        transform = transforms.Compose(
            [
                CenterCropTransform(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        dataset = UnlabeledImageDataset(args.data_path, transform=transform)
    batch_size = int(training_cfg.get("batch_size", 16))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = amp.GradScaler("cuda", enabled=(args.precision == "fp16" and device.type == "cuda"),
                            init_scale=1024)

    # ----- optional epoch-wise FID evaluation -----
    fid_enabled = bool(fid_cfg.get("enabled", False))
    fid_every = int(fid_cfg.get("every_n_epochs", 1))
    fid_num_samples = int(fid_cfg.get("num_samples", 2000))
    fid_batch_size = int(fid_cfg.get("batch_size", batch_size))
    fid_precision = str(fid_cfg.get("precision", args.precision))
    fid_use_amp = fid_precision in {"bf16", "fp16"}
    fid_amp_dtype = torch.bfloat16 if fid_precision == "bf16" else torch.float16
    fid_real_path = str(fid_cfg.get("real_path", args.data_path))
    fid_cond_path = str(fid_cfg.get("cond_path", fid_real_path))
    fid_clean_samples = bool(fid_cfg.get("clean_samples", True))
    fid_samples_find_deep = bool(fid_cfg.get("samples_find_deep", True))
    fid_save_pair_grid = bool(fid_cfg.get("save_pair_grid", True))
    fid_fixed_grid_samples = bool(fid_cfg.get("fixed_grid_samples", True))
    fid_grid_pairs_per_image = int(fid_cfg.get("grid_pairs_per_image", 8))
    fid_max_grid_pairs = int(fid_cfg.get("max_grid_pairs", 256))
    fid_save_pair_manifest = bool(fid_cfg.get("save_pair_manifest", True))
    fid_cfg_scale = float(fid_cfg.get("cfg_scale", 1.0))
    fid_cfg_t_min = float(fid_cfg.get("cfg_t_min", 0.0))
    fid_cfg_t_max = float(fid_cfg.get("cfg_t_max", 1.0))
    if fid_cfg_scale > 1.0 and not (fid_cfg_t_min < fid_cfg_t_max):
        raise ValueError("fid_eval cfg interval must satisfy cfg_t_min < cfg_t_max")

    # Build sampler for generation during evaluation.
    sampler_cfg = full_cfg.get("sampler", None)
    if sampler_cfg is not None:
        sampler_cfg = dict(OmegaConf.to_container(sampler_cfg, resolve=True))
    else:
        sampler_cfg = {"mode": "ODE", "params": {"sampling_method": "dopri5", "num_steps": 50, "atol": 1e-6, "rtol": 1e-3}}
    sampler_mode = str(sampler_cfg.get("mode", "ODE")).upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    sampler = Sampler(transport)
    if sampler_mode == "ODE":
        eval_sample_fn = sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sample_fn = sampler.sample_sde(**sampler_params)
    else:
        raise ValueError(f"Unsupported sampler mode for FID eval: {sampler_mode}")

    latent_size = misc_cfg.get("latent_size", [768, 16, 16])
    latent_size = tuple(int(v) for v in latent_size)

    cond_transform = transforms.Compose(
        [
            CenterCropTransform(args.image_size),
            transforms.ToTensor(),
        ]
    )
    cond_dataset = UnlabeledImageDataset(fid_cond_path, transform=cond_transform, return_path=True)
    cond_loader = DataLoader(
        cond_dataset,
        batch_size=fid_batch_size,
        shuffle=not fid_fixed_grid_samples,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    @torch.no_grad()
    def run_fid_eval(epoch_idx: int) -> float:
        ema_model.eval()
        rae.eval()
        conditioner.eval()

        fid_root = out_dir / "fid_samples"
        sample_dir = fid_root / f"epoch-{epoch_idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        pair_grid_dir = out_dir / "fid_pair_grids" / f"epoch-{epoch_idx:04d}"
        if fid_save_pair_grid:
            pair_grid_dir.mkdir(parents=True, exist_ok=True)

        pair_manifest_path = out_dir / "fid_pair_grids" / f"epoch-{epoch_idx:04d}_pairs.csv"
        pair_rows: list[tuple[int, str, str]] = []

        generated = 0
        img_idx = 0
        grid_idx = 0
        paired_saved = 0
        while generated < fid_num_samples:
            for images_cond, cond_paths in cond_loader:
                if generated >= fid_num_samples:
                    break
                images_cond = images_cond.to(device, non_blocking=True)
                bsz = images_cond.size(0)
                cond_paths = list(cond_paths)
                remain = fid_num_samples - generated
                if bsz > remain:
                    images_cond = images_cond[:remain]
                    cond_paths = cond_paths[:remain]
                    bsz = remain

                z0 = torch.randn(bsz, *latent_size, device=device)
                cond = conditioner(images_cond)

                if fid_cfg_scale > 1.0:
                    z = torch.cat([z0, z0], dim=0)

                    def ema_forward_with_cond_cfg(x, t, cfg_scale, cfg_interval=(0.0, 1.0), cond: torch.Tensor | None = None):
                        half = x[: len(x) // 2]
                        combined = torch.cat([half, half], dim=0)

                        if cond is None:
                            raise ValueError("cond must be provided when using fid_eval cfg_scale > 1.0")
                        if cond.shape[0] == x.shape[0]:
                            cond_half = cond[: cond.shape[0] // 2]
                        elif cond.shape[0] == x.shape[0] // 2:
                            cond_half = cond
                        else:
                            raise ValueError(f"Invalid cond batch size for CFG: cond={cond.shape[0]}, x={x.shape[0]}")

                        cond_combined = torch.cat([cond_half, torch.zeros_like(cond_half)], dim=0)

                        model_out = ema_model.forward(combined, t, cond=cond_combined)
                        eps, rest = model_out[:, : ema_model.in_channels], model_out[:, ema_model.in_channels :]
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

                    model_fn = ema_forward_with_cond_cfg
                    model_kwargs = dict(
                        cond=cond,
                        cfg_scale=fid_cfg_scale,
                        cfg_interval=(fid_cfg_t_min, fid_cfg_t_max),
                    )
                else:
                    z = z0
                    model_fn = ema_model.forward
                    model_kwargs = dict(cond=cond)

                with amp.autocast("cuda", enabled=fid_use_amp, dtype=fid_amp_dtype):
                    zhat = eval_sample_fn(z, model_fn, **model_kwargs)[-1]
                    if fid_cfg_scale > 1.0:
                        zhat = zhat[:bsz]
                    samples = rae.decode(zhat).clamp(0, 1)

                local_sample_paths: list[str] = []
                for i in range(samples.size(0)):
                    out_path = sample_dir / f"{img_idx:06d}.png"
                    save_image(samples[i], out_path, normalize=False)
                    local_sample_paths.append(str(out_path))
                    img_idx += 1

                if fid_save_pair_manifest:
                    for p, s in zip(cond_paths, local_sample_paths):
                        pair_rows.append((generated, p, s))
                        generated += 1
                else:
                    generated += bsz

                if fid_save_pair_grid and paired_saved < fid_max_grid_pairs:
                    remain_pair_budget = max(fid_max_grid_pairs - paired_saved, 0)
                    n_pairs = min(fid_grid_pairs_per_image, bsz, remain_pair_budget)
                    if n_pairs > 0:
                        pair_tensors = []
                        inputs_vis = images_cond[:n_pairs].detach().cpu()
                        outputs_vis = samples[:n_pairs].detach().cpu()
                        for i in range(n_pairs):
                            pair_tensors.append(inputs_vis[i])
                            pair_tensors.append(outputs_vis[i])
                        grid = make_grid(torch.stack(pair_tensors), nrow=2, normalize=False)
                        save_image(grid, pair_grid_dir / f"pair_grid_{grid_idx:04d}.png", normalize=False)
                        grid_idx += 1
                        paired_saved += n_pairs

        if fid_save_pair_manifest:
            pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with pair_manifest_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "condition_input_path", "generated_image_path"])
                writer.writerows(pair_rows)

        # FID vs real distribution
        metrics = calculate_metrics(
            input1=str(sample_dir),
            input2=fid_real_path,
            fid=True,
            cuda=(device.type == "cuda"),
            batch_size=min(64, fid_batch_size),
            samples_find_deep=fid_samples_find_deep,
            isc=False,
            kid=False,
            prc=False,
        )
        fid_value = float(metrics["frechet_inception_distance"])

        # FID vs conditioning images (paired FID)
        cond_ref_dir = fid_root / f"epoch-{epoch_idx:04d}_cond_ref"
        cond_ref_dir.mkdir(parents=True, exist_ok=True)
        cond_idx = 0
        for images_cond, _ in cond_loader:
            for i in range(images_cond.size(0)):
                if cond_idx >= img_idx:
                    break
                save_image(images_cond[i], cond_ref_dir / f"{cond_idx:06d}.png", normalize=False)
                cond_idx += 1
            if cond_idx >= img_idx:
                break

        metrics_paired = calculate_metrics(
            input1=str(sample_dir),
            input2=str(cond_ref_dir),
            fid=True,
            cuda=(device.type == "cuda"),
            batch_size=min(64, fid_batch_size),
            samples_find_deep=False,
            isc=False,
            kid=False,
            prc=False,
        )
        paired_fid_value = float(metrics_paired["frechet_inception_distance"])
        shutil.rmtree(cond_ref_dir, ignore_errors=True)

        if fid_save_pair_manifest and fid_clean_samples:
            print(
                f"[info] pair manifest kept at {pair_manifest_path}, generated PNGs were cleaned by clean_samples=true",
                flush=True,
            )

        if fid_clean_samples:
            shutil.rmtree(sample_dir, ignore_errors=True)

        ema_model.train()
        return fid_value, paired_fid_value

    # ----- pair grid generation (every N epochs, independent of eval method) -----
    grid_every = int(full_cfg.get("grid_eval", {}).get("every_n_epochs", 2))
    grid_max_pairs = int(fid_cfg.get("max_grid_pairs", 32))
    grid_pairs_per_batch = int(fid_cfg.get("grid_pairs_per_image", 8))

    @torch.no_grad()
    def run_grid_eval(epoch_idx: int) -> None:
        ema_model.eval()
        grid_dir = out_dir / "pair_grids" / f"epoch-{epoch_idx:04d}"
        grid_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        grid_idx = 0
        for images_cond, _ in cond_loader:
            if saved >= grid_max_pairs:
                break
            images_cond = images_cond.to(device, non_blocking=True)
            n_pairs = min(grid_pairs_per_batch, images_cond.size(0), grid_max_pairs - saved)
            images_cond = images_cond[:n_pairs]
            bsz = n_pairs

            z0 = torch.randn(bsz, *latent_size, device=device, dtype=torch.float32)
            cond = conditioner(images_cond).float()

            if fid_cfg_scale > 1.0:
                z = torch.cat([z0, z0], dim=0)

                def _cfg_fn(x, t, cfg_scale, cfg_interval=(0.0, 1.0), cond=None):
                    half = x[: len(x) // 2]
                    combined = torch.cat([half, half], dim=0)
                    cond_half = cond[: cond.shape[0] // 2] if cond.shape[0] == x.shape[0] else cond
                    cond_combined = torch.cat([cond_half, torch.zeros_like(cond_half)], dim=0)
                    model_out = ema_model.forward(combined, t, cond=cond_combined)
                    eps, rest = model_out[:, :ema_model.in_channels], model_out[:, ema_model.in_channels:]
                    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                    t_half = t[: len(t) // 2]
                    guided_eps = torch.where(
                        ((t_half >= cfg_interval[0]) & (t_half <= cfg_interval[1])).view(-1, *[1] * (len(cond_eps.shape) - 1)),
                        uncond_eps + cfg_scale * (cond_eps - uncond_eps),
                        cond_eps,
                    )
                    return torch.cat([torch.cat([guided_eps, guided_eps], dim=0), rest], dim=1)

                model_fn = _cfg_fn
                model_kwargs = dict(cond=cond, cfg_scale=fid_cfg_scale, cfg_interval=(fid_cfg_t_min, fid_cfg_t_max))
            else:
                z = z0
                model_fn = ema_model.forward
                model_kwargs = dict(cond=cond)

            with amp.autocast("cuda", enabled=False):
                zhat = eval_sample_fn(z, model_fn, **model_kwargs)[-1]
                if fid_cfg_scale > 1.0:
                    zhat = zhat[:bsz]
                samples = rae.decode(zhat).clamp(0, 1)

            pair_tensors = []
            for i in range(bsz):
                pair_tensors.append(images_cond[i].cpu())
                pair_tensors.append(samples[i].cpu())
            grid = make_grid(torch.stack(pair_tensors), nrow=2, normalize=False)
            save_image(grid, grid_dir / f"pair_grid_{grid_idx:04d}.png", normalize=False)
            grid_idx += 1
            saved += bsz

        ema_model.train()
        print(f"[epoch {epoch_idx:03d}] pair grids saved to {grid_dir} ({saved} pairs)", flush=True)

    # ----- optional epoch-wise validation loss evaluation -----
    val_enabled = bool(val_cfg.get("enabled", False))
    val_every = int(val_cfg.get("every_n_epochs", 1))
    val_path = str(val_cfg.get("val_path", args.data_path))
    val_batch_size = int(val_cfg.get("batch_size", batch_size))

    if val_enabled:
        val_transform = transforms.Compose(
            [
                CenterCropTransform(args.image_size),
                transforms.ToTensor(),
            ]
        )
        val_dataset = UnlabeledImageDataset(val_path, transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )

        @torch.no_grad()
        def run_val_eval(epoch_idx: int) -> float:
            ema_model.eval()
            total_loss = 0.0
            total_n = 0
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    z = rae.encode(images)
                    cond = conditioner(images)
                    loss = transport.training_losses(ema_model, z, dict(cond=cond))["loss"]
                total_loss += loss.sum().item()
                total_n += images.size(0)
            ema_model.train()
            return total_loss / max(total_n, 1)

    epochs = int(training_cfg.get("epochs", 50))
    log_interval = int(training_cfg.get("log_interval", 100))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    ckpt_cfg = dict(training_cfg.get("checkpoint", {}))
    ckpt_save_every = int(ckpt_cfg.get("save_every_n_epochs", 1))
    ckpt_keep_last_n = int(ckpt_cfg.get("keep_last_n", 2))
    ckpt_save_mode = str(ckpt_cfg.get("save_mode", "ema_only")).lower()
    if ckpt_save_mode not in {"full", "model_ema", "ema_only", "model_only"}:
        raise ValueError(
            "training.checkpoint.save_mode must be one of: full, model_ema, ema_only, model_only"
        )
    out_dir = Path(args.results_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * epochs
    print(
        f"[info] training_plan: epochs={epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}",
        flush=True,
    )
    print(
        f"[info] checkpoint_plan: mode={ckpt_save_mode}, save_every_n_epochs={ckpt_save_every}, keep_last_n={ckpt_keep_last_n}",
        flush=True,
    )

    resume_arg = args.resume.strip()
    resume_path: Path | None = None
    if resume_arg:
        if resume_arg.lower() == "latest":
            ckpts = _collect_epoch_checkpoints(out_dir)
            if len(ckpts) == 0:
                raise FileNotFoundError(f"--resume=latest requested but no epoch checkpoints found in {out_dir}")
            resume_path = ckpts[-1][1]
        else:
            resume_path = Path(resume_arg).expanduser().resolve()
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    start_epoch = 1
    global_step = 0
    if resume_path is not None:
        ckpt = full_torch_load(resume_path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Unexpected checkpoint type at {resume_path}: {type(ckpt)}")

        loaded_model = False
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            loaded_model = True

        if "ema" in ckpt:
            ema_model.load_state_dict(ckpt["ema"], strict=False)
            if not loaded_model:
                # ema_only 체크포인트도 학습 재개 가능하도록 model을 ema로 초기화
                model.load_state_dict(ckpt["ema"], strict=False)
                loaded_model = True

        if not loaded_model:
            raise RuntimeError(
                f"Checkpoint has neither 'model' nor 'ema' state_dict: {resume_path}"
            )

        if (not args.reset_optimizer) and ("optimizer" in ckpt):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("[info] optimizer state restored from checkpoint", flush=True)
            except Exception as e:
                print(f"[warn] failed to restore optimizer state: {e}", flush=True)
        elif "optimizer" not in ckpt:
            print("[info] checkpoint has no optimizer state; continuing with fresh optimizer", flush=True)
        else:
            print("[info] --reset-optimizer enabled; using fresh optimizer", flush=True)

        resume_epoch = int(ckpt.get("epoch", 0))
        resume_step = int(ckpt.get("step", resume_epoch * steps_per_epoch))
        start_epoch = resume_epoch + 1
        global_step = resume_step
        print(
            f"[info] resumed from {resume_path} (epoch={resume_epoch}, step={resume_step}); next epoch={start_epoch}",
            flush=True,
        )

        if start_epoch > epochs:
            print(
                f"[info] resume epoch {resume_epoch} already reached configured epochs={epochs}. Nothing to do.",
                flush=True,
            )
            return

        del ckpt
        gc.collect()

    best_val_loss = float("inf")
    session_step_offset = global_step
    train_t0 = time()
    for epoch in range(start_epoch, epochs + 1):
        loss_running = 0.0
        t0 = time()
        for step, batch in enumerate(loader, start=1):
            if use_cached_latents:
                z = batch["z"].to(device, non_blocking=True).float()
                if "cond" in batch:
                    cond = batch["cond"].to(device, non_blocking=True).float()
                else:
                    images = batch["image"].to(device, non_blocking=True)
                    with torch.no_grad():
                        cond = conditioner(images)
            else:
                images, _ = batch
                images = images.to(device, non_blocking=True)
                with torch.no_grad():
                    z = rae.encode(images)
                    cond = conditioner(images)

            model_kwargs = dict(cond=cond)
            with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = transport.training_losses(model, z, model_kwargs)["loss"].mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, decay=ema_decay)

            global_step += 1
            loss_running += float(loss.item())
            if global_step % log_interval == 0:
                avg = loss_running / log_interval
                elapsed_epoch = max(time() - t0, 1e-6)
                epoch_it_s = step / elapsed_epoch
                epoch_eta_min = (len(loader) - step) / max(epoch_it_s, 1e-6) / 60.0

                global_elapsed = max(time() - train_t0, 1e-6)
                global_done_steps = (epoch - 1) * len(loader) + step
                session_done_steps = global_done_steps - session_step_offset
                global_it_s = max(session_done_steps, 1) / global_elapsed
                total_eta_min = (total_steps - global_done_steps) / max(global_it_s, 1e-6) / 60.0

                print(
                    f"[epoch {epoch:03d} step {global_step:07d}] "
                    f"loss={avg:.6f} it/s={epoch_it_s:.2f} "
                    f"epoch_eta_min={epoch_eta_min:.1f} total_eta_min={total_eta_min:.1f}",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": avg,
                        "train/it_per_sec": epoch_it_s,
                    }, step=global_step)
                loss_running = 0.0

        elapsed = time() - t0

        # ── 체크포인트 payload 구성 ───────────────────────────────────────
        ckpt_payload = {
            "epoch": epoch,
            "step": global_step,
            "config": OmegaConf.to_container(full_cfg, resolve=True),
            "save_mode": ckpt_save_mode,
        }
        if ckpt_save_mode == "full":
            ckpt_payload.update(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            )
        elif ckpt_save_mode == "model_ema":
            ckpt_payload.update(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                }
            )
        elif ckpt_save_mode == "model_only":
            ckpt_payload.update({"model": model.state_dict()})
        else:
            ckpt_payload.update({"ema": ema_model.state_dict()})

        # latest 항상 저장
        torch.save(ckpt_payload, out_dir / "latest.pt")

        # 5 epoch 단위 저장
        should_save_ckpt = (epoch % max(ckpt_save_every, 1) == 0) or (epoch == epochs)
        if should_save_ckpt:
            ckpt_path = out_dir / f"ep-{epoch:04d}.pt"
            torch.save(ckpt_payload, ckpt_path)
            print(f"[epoch {epoch:03d}] done in {elapsed/60:.1f} min, saved: {ckpt_path}", flush=True)
            _prune_old_checkpoints(out_dir, ckpt_keep_last_n)
        else:
            print(
                f"[epoch {epoch:03d}] done in {elapsed/60:.1f} min, checkpoint skipped (save_every_n_epochs={ckpt_save_every})",
                flush=True,
            )

        if fid_enabled and (epoch % max(fid_every, 1) == 0):
            fid_t0 = time()
            fid_value, paired_fid_value = run_fid_eval(epoch)
            print(
                f"[epoch {epoch:03d}] FID={fid_value:.4f} paired_FID={paired_fid_value:.4f} "
                f"(num_samples={fid_num_samples}, real_path={fid_real_path}) "
                f"eval_time_min={(time() - fid_t0)/60:.1f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "eval/FID": fid_value,
                    "eval/paired_FID": paired_fid_value,
                    "epoch": epoch,
                }, step=global_step)

        if val_enabled and (epoch % max(val_every, 1) == 0):
            val_t0 = time()
            val_loss = run_val_eval(epoch)
            print(
                f"[epoch {epoch:03d}] val_loss={val_loss:.6f} "
                f"(val_path={val_path}) eval_time_min={(time() - val_t0)/60:.1f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({"eval/val_loss": val_loss, "epoch": epoch}, step=global_step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt_payload, out_dir / "best.pt")
                print(f"[epoch {epoch:03d}] best val_loss={best_val_loss:.6f}, saved: best.pt", flush=True)

        if epoch % max(grid_every, 1) == 0:
            grid_t0 = time()
            run_grid_eval(epoch)
            print(f"[epoch {epoch:03d}] grid_eval_time_min={(time() - grid_t0)/60:.1f}", flush=True)


    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

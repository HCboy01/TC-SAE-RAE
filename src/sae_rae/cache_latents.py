#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import amp
from torch.utils.data import DataLoader
from torchvision import transforms

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sae_rae.script_utils import RecursiveImageDataset, add_sys_path


class CenterCropTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image):
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


def build_loader(
    root: str,
    image_size: int,
    batch_size: int,
    workers: int,
    *,
    hflip: bool,
    files: list[Path] | None = None,
) -> tuple[RecursiveImageDataset, DataLoader]:
    dataset = RecursiveImageDataset(
        root,
        transform=OfflineAugTransform(image_size, hflip=hflip),
        files=files,
        return_path=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute RAE z latents and/or SAE conditioning latents.")
    parser.add_argument("--config", type=str, required=True, help="SAE-RAE training YAML config")
    parser.add_argument("--data-path", type=str, required=True, help="Image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write cache arrays, metadata.json, and paths.jsonl")
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--storage-dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="z_cond",
        choices=["z_cond", "z", "cond"],
        help="Which arrays to cache. Use 'z' to skip SAE conditioner outputs.",
    )
    parser.add_argument(
        "--num-augs",
        type=int,
        default=2,
        help="Offline copies per image. aug 0 is original; odd aug ids are horizontal flips.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_augs < 1:
        raise ValueError("--num-augs must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script currently expects CUDA.")

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    add_sys_path(project_root / "vendor" / "rae_src")
    add_sys_path(project_root / "src")

    from stage1 import RAE
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs
    from sae_rae.conditioning import DinoClsSaeConditioner

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir is not empty: {out_dir}. Pass --overwrite to replace cache files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = OmegaConf.load(args.config)
    rae_config, _model_config, _transport_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("Config must contain stage_1.")

    cache_z = args.cache_mode in {"z_cond", "z"}
    cache_cond = args.cache_mode in {"z_cond", "cond"}

    rae = None
    if cache_z:
        rae = instantiate_from_config(rae_config).to(device)
        rae.eval()
        rae.requires_grad_(False)

    conditioner = None
    if cache_cond:
        sae_cond_cfg = full_cfg.get("sae_condition", None)
        if sae_cond_cfg is None:
            raise ValueError("Config must include top-level `sae_condition` block when caching cond.")

        s1_params = rae_config.get("params", {})
        encoder_params = s1_params.get("encoder_params", {})
        conditioner = DinoClsSaeConditioner(
            encoder_config_path=str(sae_cond_cfg.get("encoder_config_path", s1_params.get("encoder_config_path"))),
            dinov2_path=str(sae_cond_cfg.get("dinov2_path", encoder_params.get("dinov2_path", s1_params.get("encoder_config_path")))),
            encoder_input_size=int(sae_cond_cfg.get("encoder_input_size", s1_params.get("encoder_input_size", 224))),
            sae_ckpt_path=str(sae_cond_cfg.get("sae_ckpt")),
            sae_src_path=str(sae_cond_cfg.get("sae_src_path", str(project_root / "src"))),
        ).to(device)
        conditioner.eval()
        conditioner.requires_grad_(False)

    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    np_dtype = np.float16 if args.storage_dtype == "fp16" else np.float32

    shared_files = RecursiveImageDataset.collect_files(Path(args.data_path))
    probe_dataset, probe_loader = build_loader(
        args.data_path,
        args.image_size,
        args.batch_size,
        args.workers,
        hflip=False,
        files=shared_files,
    )
    total = len(probe_dataset) * args.num_augs

    first_images, _ = next(iter(probe_loader))
    first_images = first_images.to(device, non_blocking=True)
    with torch.no_grad(), amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        first_z = rae.encode(first_images) if cache_z else None
        first_cond = conditioner(first_images) if cache_cond else None

    z_shape = tuple(first_z.shape[1:]) if first_z is not None else None
    cond_shape = tuple(first_cond.shape[1:]) if first_cond is not None else None
    z_mm = None
    cond_mm = None
    if z_shape is not None:
        z_mm = np.lib.format.open_memmap(out_dir / "z.npy", mode="w+", dtype=np_dtype, shape=(total, *z_shape))
    if cond_shape is not None:
        cond_mm = np.lib.format.open_memmap(out_dir / "cond.npy", mode="w+", dtype=np_dtype, shape=(total, *cond_shape))

    metadata = {
        "format": "sae_rae_latent_cache_v1",
        "cache_mode": args.cache_mode,
        "source_data_path": str(Path(args.data_path).resolve()),
        "config": str(Path(args.config).resolve()),
        "image_size": args.image_size,
        "num_source_images": len(probe_dataset),
        "num_augs": args.num_augs,
        "num_samples": total,
        "z_shape": [total, *z_shape] if z_shape is not None else None,
        "cond_shape": [total, *cond_shape] if cond_shape is not None else None,
        "storage_dtype": args.storage_dtype,
        "precision": args.precision,
        "augmentation": "center_crop + deterministic horizontal flip on odd aug ids",
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    paths_f = (out_dir / "paths.jsonl").open("w")
    t0 = time()
    write_idx = 0
    try:
        for aug_idx in range(args.num_augs):
            hflip = (aug_idx % 2) == 1
            dataset, loader = build_loader(
                args.data_path,
                args.image_size,
                args.batch_size,
                args.workers,
                hflip=hflip,
                files=shared_files,
            )

            for step, (images, paths) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                with torch.no_grad(), amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    z = rae.encode(images) if cache_z else None
                    cond = conditioner(images) if cache_cond else None

                bsz = images.shape[0]
                if z_mm is not None:
                    z_mm[write_idx: write_idx + bsz] = z.detach().float().cpu().numpy().astype(np_dtype, copy=False)
                if cond_mm is not None:
                    cond_mm[write_idx: write_idx + bsz] = cond.detach().float().cpu().numpy().astype(np_dtype, copy=False)
                for path in paths:
                    paths_f.write(json.dumps({"path": path, "aug_idx": aug_idx, "hflip": hflip}) + "\n")
                write_idx += bsz

                if step % 50 == 0:
                    done = write_idx
                    it_s = done / max(time() - t0, 1e-6)
                    eta_min = (total - done) / max(it_s, 1e-6) / 60.0
                    print(
                        f"[cache] aug={aug_idx + 1}/{args.num_augs} step={step:05d} "
                        f"written={done}/{total} ({100.0 * done / total:.1f}%) eta={eta_min:.1f}m",
                        flush=True,
                    )
    finally:
        paths_f.close()
        if z_mm is not None:
            z_mm.flush()
        if cond_mm is not None:
            cond_mm.flush()

    if write_idx != total:
        raise RuntimeError(f"Cache write count mismatch: wrote {write_idx}, expected {total}")

    elapsed = time() - t0
    total_bytes = 0
    if z_mm is not None:
        total_bytes += math.prod(z_mm.shape) * z_mm.dtype.itemsize
    if cond_mm is not None:
        total_bytes += math.prod(cond_mm.shape) * cond_mm.dtype.itemsize
    gb = total_bytes / 1e9
    print(
        f"[done] wrote {total} samples to {out_dir} in {elapsed / 60.0:.1f}m "
        f"({gb:.2f} GB, dtype={args.storage_dtype})",
        flush=True,
    )


if __name__ == "__main__":
    main()

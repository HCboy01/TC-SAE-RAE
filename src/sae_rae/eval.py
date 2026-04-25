#!/usr/bin/env python3
"""Evaluate the base DiT model (no SAE fine-tuning) on the validation set.

Usage:
    python src/sae_rae/eval.py \
        --config configs/sae_rae/ImageNet256/DiTDH-XL_DINOv2-B_SAECLS_ep10_nonorm.yaml \
        --base-ckpt /home/gimhyeongchan97/RAE/models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt \
        --val-path /home/gimhyeongchan97/datasets/ffhq256/imagefolder/val \
        --batch-size 8 --image-size 256
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import amp
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sae_rae.script_utils import RecursiveImageDataset, add_sys_path, safe_torch_load


def center_crop_arr(pil_image, image_size):
    w, h = pil_image.size
    crop = min(w, h)
    pil_image = pil_image.crop(
        ((w - crop) // 2, (h - crop) // 2,
         (w + crop) // 2, (h + crop) // 2)
    )
    return pil_image.resize((image_size, image_size), Image.BICUBIC)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--base-ckpt", required=True, help="Path to base DiT checkpoint (stage2_model.pt)")
    p.add_argument("--val-path", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- sys path setup (same as sae_rae/train.py) ---
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    add_sys_path(project_root / "vendor" / "rae_src")
    add_sys_path(src_dir)

    from stage1 import RAE
    from stage2.transport import create_transport
    from utils.model_utils import instantiate_from_config

    # --- Config ---
    full_cfg = OmegaConf.load(args.config)
    rae_config    = full_cfg.get("stage_1")
    model_config  = OmegaConf.to_container(full_cfg.get("stage_2"), resolve=True)
    transport_cfg = full_cfg.get("transport")
    misc_cfg      = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True))

    # --- RAE encoder ---
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()
    rae.requires_grad_(False)
    print("RAE loaded.")

    # --- Base DiT model (cond_dim=0 → no SAE conditioning head at all) ---
    model_config_base = dict(model_config)
    model_config_base.pop("ckpt", None)
    model_config_base["params"] = dict(model_config_base.get("params", {}))
    model_config_base["params"]["cond_dim"] = 0   # no SAE cond

    model = instantiate_from_config(OmegaConf.create(model_config_base)).to(device)

    # Load base checkpoint
    state_dict = safe_torch_load(args.base_ckpt, map_location="cpu")
    if "ema" in state_dict:
        state_dict = state_dict["ema"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    keys = model.load_state_dict(state_dict, strict=False)
    if keys.missing_keys:
        print(f"[info] missing keys: {keys.missing_keys}")
    if keys.unexpected_keys:
        print(f"[warn] unexpected keys: {keys.unexpected_keys}")
    model.eval()
    model.requires_grad_(False)
    print(f"Base DiT loaded from: {args.base_ckpt}")

    # --- Transport ---
    transport_params = dict(OmegaConf.to_container(transport_cfg.get("params", {}), resolve=True))
    shift_dim  = int(misc_cfg.get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(misc_cfg.get("time_dist_shift_base", 4096))
    transport_params.pop("time_dist_shift", None)
    transport_params["time_dist_shift"] = math.sqrt(shift_dim / shift_base)
    transport = create_transport(**transport_params)

    # --- Val dataset ---
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = RecursiveImageDataset(args.val_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    print(f"Val set: {len(dataset)} images  ({len(loader)} batches)")

    # --- Evaluation ---
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                z = rae.encode(images)
                # cond=None → no SAE conditioning (base model behavior)
                loss = transport.training_losses(model, z, dict(cond=None))["loss"]
            total_loss += loss.sum().item()
            total_n += images.size(0)
            if (i + 1) % 10 == 0:
                print(f"  batch {i+1}/{len(loader)}  running_loss={total_loss/total_n:.6f}")

    val_loss = total_loss / max(total_n, 1)
    print(f"\n=== Base model val loss: {val_loss:.6f} (n={total_n}) ===")


if __name__ == "__main__":
    main()

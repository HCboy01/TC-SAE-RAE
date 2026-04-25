#!/usr/bin/env python3
"""Pre-extract DINOv2 CLS tokens from an image dataset and save to disk.

Usage:
    python src/tc_sae/cache_features.py \
        --config configs/tc_sae/ffhq256_sae_tc_preact_v8.yaml \
        --output-dir /scratch/x3411a10/unconditional_diffusion/SAE-DINO/features/ffhq256 \
        [--batch-size 256] [--workers 8]

Saves per split:
    <output-dir>/train_features.bin        — float32 memmap (N, d_model)
    <output-dir>/train_features_shape.npy  — shape [N, d_model]
    <output-dir>/train_paths.txt           — one absolute image path per line
    (same for eval_ if eval.enabled=true)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2WithRegistersModel

try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError("pip install omegaconf")


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImagePathDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.files = sorted(
            p for p in Path(root).rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not self.files:
            raise RuntimeError(f"No images found under: {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, str(path)


def build_extractor(rae_config_path: str, device: torch.device):
    cfg = OmegaConf.load(rae_config_path)
    params = cfg.stage_1.params
    encoder_config_path = str(params.get("encoder_config_path", "facebook/dinov2-with-registers-base"))
    encoder_input_size = int(params.get("encoder_input_size", 224))
    encoder_params = params.get("encoder_params", {})
    dinov2_path = str(encoder_params.get("dinov2_path", encoder_config_path))

    proc = AutoImageProcessor.from_pretrained(encoder_config_path)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(device)

    encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path).to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    input_size = encoder_input_size
    hidden_size = int(encoder.config.hidden_size)

    @torch.inference_mode()
    def extract(images: torch.Tensor) -> torch.Tensor:
        _, _, h, w = images.shape
        if h != input_size or w != input_size:
            images = nn.functional.interpolate(
                images, size=(input_size, input_size),
                mode="bicubic", align_corners=False,
            )
        x = (images - mean) / std
        return encoder(x).last_hidden_state[:, 0, :]

    return extract, hidden_size


def extract_and_save(
    data_path: str,
    split_name: str,
    output_dir: Path,
    extractor,
    hidden_size: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> None:
    features_path = output_dir / f"{split_name}_features.bin"
    shape_path = output_dir / f"{split_name}_features_shape.npy"
    paths_path = output_dir / f"{split_name}_paths.txt"

    if features_path.exists() and shape_path.exists() and paths_path.exists():
        print(f"[skip] {split_name} cache already exists: {features_path}", flush=True)
        return

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    dataset = ImagePathDataset(data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    n = len(dataset)
    size_gb = n * hidden_size * 4 / 1e9
    print(f"[{split_name}] {n} images → {features_path} ({size_gb:.2f} GB)", flush=True)

    features = np.memmap(features_path, dtype=np.float32, mode="w+", shape=(n, hidden_size))
    all_paths: list[str] = []
    idx = 0

    for batch_idx, (images, batch_paths) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        cls = extractor(images).float().cpu().numpy()
        bsz = cls.shape[0]
        features[idx: idx + bsz] = cls
        features.flush()
        all_paths.extend(batch_paths)
        idx += bsz
        if (batch_idx + 1) % 100 == 0:
            print(f"  [{split_name}] {idx}/{n} ({100 * idx / n:.1f}%)", flush=True)

    del features
    np.save(str(shape_path), np.array([idx, hidden_size], dtype=np.int64))
    paths_path.write_text("\n".join(all_paths))
    print(f"[{split_name}] done — {idx} features saved → {features_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-extract DINOv2 CLS tokens")
    parser.add_argument("--config", type=str, required=True, help="tc_sae training config YAML")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save .bin and .npy files")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    path_cfg = cfg.paths
    data_cfg = cfg.data
    eval_cfg = cfg.get("eval", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # resolve rae_config relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    rae_cfg_path = Path(str(path_cfg.rae_config))
    if not rae_cfg_path.is_absolute():
        rae_cfg_path = repo_root / rae_cfg_path

    print(f"[info] device={device}", flush=True)
    extractor, hidden_size = build_extractor(str(rae_cfg_path), device)
    print(f"[info] d_model={hidden_size}", flush=True)

    extract_and_save(
        data_path=str(path_cfg.data_path),
        split_name="train",
        output_dir=output_dir,
        extractor=extractor,
        hidden_size=hidden_size,
        image_size=int(data_cfg.image_size),
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=device,
    )

    if eval_cfg.get("enabled", False):
        val_path = str(eval_cfg.get("data_path", ""))
        if val_path:
            extract_and_save(
                data_path=val_path,
                split_name="eval",
                output_dir=output_dir,
                extractor=extractor,
                hidden_size=hidden_size,
                image_size=int(data_cfg.image_size),
                batch_size=args.batch_size,
                num_workers=args.workers,
                device=device,
            )


if __name__ == "__main__":
    main()

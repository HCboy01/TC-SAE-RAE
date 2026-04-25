#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision import transforms

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sae_rae.conditioning import DinoClsSaeConditioner
from sae_rae.script_utils import RecursiveImageDataset, add_sys_path


@dataclass
class TopImageRecord:
    feature_idx: int
    rank: int
    dataset_idx: int
    activation: float
    path: str
    aug_idx: int = 0
    hflip: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label SAE features from a validation dataset with top-k image grids and optional Qwen-VL / CLIP verification."
    )
    parser.add_argument("--config", type=str, required=True, help="SAE-RAE YAML config with `sae_condition`.")
    parser.add_argument("--data-path", type=str, default=None, help="Validation image directory. Required unless --cond-cache-dir is used.")
    parser.add_argument("--cond-cache-dir", type=str, default=None, help="Optional latent cache dir from src/sae_rae/cache_latents.py.")
    parser.add_argument("--output-dir", type=str, required=True, help="Where manifests, grids, labels, and metrics are written.")
    parser.add_argument("--image-size", type=int, default=256, choices=[224, 256, 512])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--top-images-per-feature", type=int, default=16)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=None, help="Optionally restrict to the first N features.")
    parser.add_argument("--feature-start", type=int, default=0)
    parser.add_argument("--min-activation", type=float, default=0.0, help="Ignore top-k entries below this activation.")
    parser.add_argument("--save-activation-matrix", action="store_true", help="Save full [num_images, num_features] activation matrix as .npy.")
    parser.add_argument("--device", type=str, default=None, help="cuda, cuda:0, or cpu. Defaults to CUDA if available.")
    parser.add_argument("--qwen-model", type=str, default=None, help="Optional Qwen-VL model id/path loaded with vLLM.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument("--qwen-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--qwen-max-model-len", type=int, default=8192)
    parser.add_argument("--qwen-max-num-seqs", type=int, default=4)
    parser.add_argument("--qwen-max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--qwen-enforce-eager", action="store_true")
    parser.add_argument("--clip-model", type=str, default=None, help="Optional CLIP model id/path for verification.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def center_crop_transform(image_size: int):
    def _crop(pil_image: Image.Image) -> torch.Tensor:
        s = image_size
        while min(*pil_image.size) >= 2 * s:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        scale = s / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
        arr = np.array(pil_image)
        cy = (arr.shape[0] - s) // 2
        cx = (arr.shape[1] - s) // 2
        return transforms.ToTensor()(Image.fromarray(arr[cy: cy + s, cx: cx + s]))

    return _crop


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_project_imports() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    add_sys_path(project_root / "vendor" / "rae_src")
    add_sys_path(project_root / "src")
    return project_root


def load_conditioner_from_config(config_path: str, device: torch.device) -> DinoClsSaeConditioner:
    from omegaconf import OmegaConf

    full_cfg = OmegaConf.load(config_path)
    sae_cond_cfg = full_cfg.get("sae_condition")
    if sae_cond_cfg is None:
        raise ValueError("Config must include top-level `sae_condition`.")

    rae_cfg = full_cfg.get("stage_1")
    if rae_cfg is None:
        raise ValueError("Config must include top-level `stage_1`.")

    s1_params = rae_cfg.get("params", {})
    encoder_params = s1_params.get("encoder_params", {})
    project_root = Path(__file__).resolve().parent.parent
    conditioner = DinoClsSaeConditioner(
        encoder_config_path=str(sae_cond_cfg.get("encoder_config_path", s1_params.get("encoder_config_path"))),
        dinov2_path=str(sae_cond_cfg.get("dinov2_path", encoder_params.get("dinov2_path", s1_params.get("encoder_config_path")))),
        encoder_input_size=int(sae_cond_cfg.get("encoder_input_size", s1_params.get("encoder_input_size", 224))),
        sae_ckpt_path=str(sae_cond_cfg.get("sae_ckpt")),
        sae_src_path=str(sae_cond_cfg.get("sae_src_path", str(project_root / "src"))),
    ).to(device)
    conditioner.eval().requires_grad_(False)
    return conditioner


def load_cond_cache(cache_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cond_path = cache_dir / "cond.npy"
    paths_path = cache_dir / "paths.jsonl"
    if not cond_path.exists():
        raise FileNotFoundError(f"Missing cond cache: {cond_path}")
    if not paths_path.exists():
        raise FileNotFoundError(f"Missing paths manifest: {paths_path}")
    cond = np.load(cond_path, mmap_mode="r")
    records: list[dict[str, Any]] = []
    with paths_path.open() as f:
        for line in f:
            records.append(json.loads(line))
    if len(records) != cond.shape[0]:
        raise RuntimeError(f"paths.jsonl size mismatch: {len(records)} rows vs cond.npy {cond.shape[0]} samples")
    return cond, records


def compute_cond_activations(
    conditioner: DinoClsSaeConditioner,
    data_path: str,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dataset = RecursiveImageDataset(
        data_path,
        transform=center_crop_transform(image_size),
        return_path=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    amp_enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    cond_chunks: list[np.ndarray] = []
    path_records: list[dict[str, Any]] = []

    with torch.no_grad():
        for step, (images, paths) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                cond = conditioner(images)
            cond_chunks.append(cond.detach().float().cpu().numpy())
            path_records.extend({"path": path, "aug_idx": 0, "hflip": False} for path in paths)
            if step % 50 == 0:
                print(f"[activations] step={step:05d} written={len(path_records)}", flush=True)

    activations = np.concatenate(cond_chunks, axis=0)
    return activations, path_records


def select_feature_slice(num_features: int, feature_start: int, max_features: int | None) -> range:
    start = max(0, feature_start)
    stop = num_features if max_features is None else min(num_features, start + max_features)
    if start >= stop:
        raise ValueError(f"Empty feature slice: start={start}, stop={stop}, num_features={num_features}")
    return range(start, stop)


def topk_for_feature(
    activations: np.ndarray,
    records: list[dict[str, Any]],
    feature_idx: int,
    top_k: int,
    min_activation: float,
) -> list[TopImageRecord]:
    column = activations[:, feature_idx]
    if top_k >= len(column):
        candidate_idx = np.argsort(-column)
    else:
        partial = np.argpartition(column, -top_k)[-top_k:]
        candidate_idx = partial[np.argsort(-column[partial])]

    results: list[TopImageRecord] = []
    for rank, dataset_idx in enumerate(candidate_idx.tolist(), start=1):
        value = float(column[dataset_idx])
        if value < min_activation:
            continue
        meta = records[dataset_idx]
        results.append(
            TopImageRecord(
                feature_idx=feature_idx,
                rank=rank,
                dataset_idx=dataset_idx,
                activation=value,
                path=str(meta["path"]),
                aug_idx=int(meta.get("aug_idx", 0)),
                hflip=bool(meta.get("hflip", False)),
            )
        )
    return results


def load_and_prepare_tile(path: str, tile_size: int = 256) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
    image.thumbnail((tile_size, tile_size), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
    offset = ((tile_size - image.width) // 2, (tile_size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def make_feature_grid(records: list[TopImageRecord], out_path: Path, grid_cols: int = 4, tile_size: int = 256) -> None:
    rows = max(1, math.ceil(max(len(records), 1) / grid_cols))
    grid = Image.new("RGB", (grid_cols * tile_size, rows * tile_size), color=(248, 248, 248))
    draw = ImageDraw.Draw(grid)
    for i, record in enumerate(records):
        tile = load_and_prepare_tile(record.path, tile_size=tile_size)
        x = (i % grid_cols) * tile_size
        y = (i // grid_cols) * tile_size
        grid.paste(tile, (x, y))
        draw.rectangle((x, y, x + tile_size - 1, y + 24), fill=(0, 0, 0))
        label = f"#{record.rank} act={record.activation:.3f}"
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def build_qwen_prompt(feature_idx: int, records: list[TopImageRecord]) -> str:
    activations = [f"{r.rank}:{r.activation:.4f}" for r in records]
    prompt = f"""
You are labeling one SAE feature from a sparse autoencoder.
The attached image is a 4x4 grid of the top activating images for feature {feature_idx}.
Top activation values by rank: {", ".join(activations)}

Return exactly one JSON object with this schema:
{{
  "feature_idx": <int>,
  "short_label": "<2-6 word concept label>",
  "confidence": <float 0 to 1>,
  "is_labelable": <true or false>,
  "evidence": ["<short phrase>", "<short phrase>"],
  "reasoning": "<1-3 sentences grounded in repeated visual pattern>",
  "failure_mode": "<empty string if labelable, else why not>"
}}

Rules:
- Focus on the repeated concept across images, not a single image.
- Prefer concrete visual concepts.
- If the images are too heterogeneous, set is_labelable=false.
- Output JSON only.
""".strip()
    return prompt


class QwenLabeler:
    def __init__(
        self,
        model_name: str,
        allowed_local_media_path: Path,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        enforce_eager: bool,
    ):
        from vllm import LLM

        self.allowed_local_media_path = allowed_local_media_path.resolve()
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt={"image": 1},
            allowed_local_media_path=str(self.allowed_local_media_path),
        )

    def label(
        self,
        grid_path: Path,
        feature_idx: int,
        records: list[TopImageRecord],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        prompt = build_qwen_prompt(feature_idx, records)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"file://{grid_path.resolve()}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        generated_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return {
            "feature_idx": feature_idx,
            "grid_path": str(grid_path),
            "raw_text": generated_text,
        }


def extract_first_json_block(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


class ClipVerifier:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import AutoProcessor, CLIPModel

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def verify(self, label: str, image_paths: list[str]) -> dict[str, Any]:
        images: list[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        text = [f"a photo of {label}"]
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = F.normalize(outputs.image_embeds, dim=-1)
            text_embeds = F.normalize(outputs.text_embeds, dim=-1)
            sims = image_embeds @ text_embeds.T

        scores = sims[:, 0].detach().float().cpu().tolist()
        return {
            "label": label,
            "mean_similarity": float(np.mean(scores)),
            "min_similarity": float(np.min(scores)),
            "max_similarity": float(np.max(scores)),
            "per_image_similarity": [float(v) for v in scores],
        }


def aggregate_metrics(parsed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(parsed_rows)
    labelable = [row for row in parsed_rows if row.get("is_labelable")]
    labels = [str(row.get("short_label", "")).strip() for row in labelable if str(row.get("short_label", "")).strip()]
    unique_labels = sorted(set(labels))
    confidences = [float(row["confidence"]) for row in labelable if row.get("confidence") is not None]
    clip_scores = [float(row["clip_mean_similarity"]) for row in labelable if row.get("clip_mean_similarity") is not None]

    return {
        "num_features_processed": total,
        "num_labelable": len(labelable),
        "labelable_ratio": (len(labelable) / total) if total else 0.0,
        "num_unique_labels": len(unique_labels),
        "label_diversity_ratio": (len(unique_labels) / len(labelable)) if labelable else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "mean_clip_similarity": float(np.mean(clip_scores)) if clip_scores else None,
    }


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output dir is not empty: {path}. Pass --overwrite to reuse it.")
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.cond_cache_dir is None and args.data_path is None:
        raise ValueError("Provide either --data-path or --cond-cache-dir.")

    out_dir = Path(args.output_dir)
    ensure_output_dir(out_dir, overwrite=args.overwrite)
    prepare_project_imports()
    device = resolve_device(args.device)

    if args.cond_cache_dir is not None:
        activations_mm, path_records = load_cond_cache(Path(args.cond_cache_dir))
        activations = np.asarray(activations_mm, dtype=np.float32)
        print(f"[load] cond cache: {activations.shape}", flush=True)
    else:
        conditioner = load_conditioner_from_config(args.config, device)
        activations, path_records = compute_cond_activations(
            conditioner=conditioner,
            data_path=str(args.data_path),
            image_size=args.image_size,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            precision=args.precision,
        )
        print(f"[load] computed activations: {activations.shape}", flush=True)

    if args.save_activation_matrix:
        np.save(out_dir / "activation_matrix.npy", activations)

    feature_range = select_feature_slice(
        num_features=activations.shape[1],
        feature_start=args.feature_start,
        max_features=args.max_features,
    )
    qwen_labeler = (
        QwenLabeler(
            model_name=args.qwen_model,
            allowed_local_media_path=out_dir,
            tensor_parallel_size=args.qwen_tensor_parallel_size,
            gpu_memory_utilization=args.qwen_gpu_memory_utilization,
            max_model_len=args.qwen_max_model_len,
            max_num_seqs=args.qwen_max_num_seqs,
            max_num_batched_tokens=args.qwen_max_num_batched_tokens,
            enforce_eager=args.qwen_enforce_eager,
        )
        if args.qwen_model is not None
        else None
    )
    clip_verifier = ClipVerifier(args.clip_model, device) if args.clip_model is not None else None

    topk_rows: list[dict[str, Any]] = []
    raw_label_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []

    for feature_idx in feature_range:
        feature_records = topk_for_feature(
            activations=activations,
            records=path_records,
            feature_idx=feature_idx,
            top_k=args.top_images_per_feature,
            min_activation=args.min_activation,
        )
        topk_rows.extend(asdict(rec) for rec in feature_records)

        grid_path = out_dir / "grids" / f"feature_{feature_idx:05d}.png"
        make_feature_grid(feature_records, grid_path, grid_cols=args.grid_cols)

        raw_row: dict[str, Any] = {
            "feature_idx": feature_idx,
            "grid_path": str(grid_path),
            "top_images": [asdict(rec) for rec in feature_records],
            "raw_text": None,
        }
        if qwen_labeler is not None:
            raw_row = qwen_labeler.label(
                grid_path=grid_path,
                feature_idx=feature_idx,
                records=feature_records,
                max_new_tokens=args.qwen_max_new_tokens,
            )
            raw_row["top_images"] = [asdict(rec) for rec in feature_records]

        raw_label_rows.append(raw_row)

        parsed_row: dict[str, Any] = {
            "feature_idx": feature_idx,
            "grid_path": str(grid_path),
            "num_top_images": len(feature_records),
            "top_activation_mean": float(np.mean([rec.activation for rec in feature_records])) if feature_records else None,
            "short_label": None,
            "confidence": None,
            "is_labelable": None,
            "evidence": None,
            "reasoning": None,
            "failure_mode": None,
            "clip_mean_similarity": None,
            "parse_error": None,
        }

        if raw_row.get("raw_text"):
            try:
                parsed = extract_first_json_block(str(raw_row["raw_text"]))
                parsed_row.update(parsed)
            except Exception as exc:
                parsed_row["parse_error"] = str(exc)

        if clip_verifier is not None and parsed_row.get("short_label") and feature_records:
            clip_stats = clip_verifier.verify(
                label=str(parsed_row["short_label"]),
                image_paths=[rec.path for rec in feature_records],
            )
            parsed_row["clip_mean_similarity"] = clip_stats["mean_similarity"]
            parsed_row["clip_stats"] = clip_stats

        parsed_rows.append(parsed_row)
        if (feature_idx - feature_range.start + 1) % 20 == 0:
            print(
                f"[feature] processed={feature_idx - feature_range.start + 1}/{len(feature_range)} last_feature={feature_idx}",
                flush=True,
            )

    write_jsonl(out_dir / "topk_manifest.jsonl", topk_rows)
    write_jsonl(out_dir / "labels_raw.jsonl", raw_label_rows)
    write_jsonl(out_dir / "labels_parsed.jsonl", parsed_rows)

    metrics = aggregate_metrics(parsed_rows)
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    run_config = vars(args).copy()
    run_config["device_resolved"] = str(device)
    with (out_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

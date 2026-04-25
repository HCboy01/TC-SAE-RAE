#!/usr/bin/env python3
"""Single-image inference: extract SAE latent and generate conditioned image.

Usage:
    python src/sae_rae/infer.py \
        --image /path/to/image.jpg \
        --ckpt  ckpts/ft_sae_cls_batch_topk_ep50/ep-0017.pt \
        --config configs/sae_rae/ImageNet256/DiTDH-XL_DINOv2-B_SAECLS.yaml \
        [--cfg-scale 2.0] [--out infer_out] [--top-k-show 40] [--seed 42]

Outputs (saved next to --out prefix):
    {out}_comparison.png  — original (left) | generated (right)
    {out}_latent.png      — bar chart of top active SAE features
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch import amp
from torchvision import transforms
from torchvision.utils import save_image

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sae_rae.script_utils import add_sys_path, full_torch_load, safe_torch_load


# ---------------------------------------------------------------------------
def _collect_epoch_checkpoints(out_dir: Path):
    pattern = re.compile(r"^ep-(\d{4})\.pt$")
    ckpts = []
    for p in out_dir.glob("ep-*.pt"):
        m = pattern.match(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


# ---------------------------------------------------------------------------
# visualisation (PIL only, no matplotlib)
# ---------------------------------------------------------------------------

def make_latent_chart(latent: torch.Tensor, top_k: int = 40) -> Image.Image:
    """Draw a horizontal bar chart of the top-k active SAE features."""
    latent = latent.float().cpu()
    nonzero = (latent != 0).sum().item()
    total = latent.numel()

    k = min(top_k, nonzero)
    if k == 0:
        img = Image.new("RGB", (800, 100), (245, 245, 245))
        ImageDraw.Draw(img).text((10, 40), "No active features.", fill=(80, 80, 80))
        return img

    values, indices = torch.topk(latent, k)
    values = values.tolist()
    indices = indices.tolist()
    max_val = max(abs(v) for v in values) or 1.0

    # layout
    bar_h = 18
    gap = 4
    label_w = 70       # feature index label
    bar_max_w = 500
    val_label_w = 80
    margin_x = 12
    margin_y = 40      # top: title
    footer = 24
    width = margin_x + label_w + bar_max_w + val_label_w + margin_x
    height = margin_y + k * (bar_h + gap) + footer

    img = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    title = f"SAE latent — top {k} active features  ({nonzero}/{total} nonzero, {100*nonzero/total:.2f}%)"
    draw.text((margin_x, 10), title, fill=(30, 30, 30), font=font_sm)

    for i, (val, idx) in enumerate(zip(values, indices)):
        y = margin_y + i * (bar_h + gap)
        # label
        draw.text((margin_x, y + 2), f"{idx:5d}", fill=(60, 60, 60), font=font)
        # bar
        bar_x = margin_x + label_w
        bar_w = int(abs(val) / max_val * bar_max_w)
        color = (70, 130, 200) if val >= 0 else (200, 90, 70)
        draw.rectangle([bar_x, y, bar_x + bar_w, y + bar_h], fill=color)
        # value label
        draw.text((bar_x + bar_max_w + 4, y + 2), f"{val:.3f}", fill=(60, 60, 60), font=font)

    draw.text(
        (margin_x, height - footer + 4),
        f"sparsity: {100*(total-nonzero)/total:.1f}%   max_val: {max_val:.4f}",
        fill=(100, 100, 100),
        font=font_sm,
    )
    return img


def make_comparison(originals: list[torch.Tensor], generated_list: list[torch.Tensor], size: int = 256) -> Image.Image:
    """Grid comparison: each row is [original | generated] for one image."""
    def to_pil(t: torch.Tensor) -> Image.Image:
        t = t.float().cpu().clamp(0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
        return Image.fromarray(arr)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    n = len(originals)
    pad = 8
    label_h = 22
    row_h = size + pad
    w = size * 2 + pad * 3
    h = label_h + pad + n * row_h + pad

    canvas = Image.new("RGB", (w, h), (230, 230, 230))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad + size // 2 - 30, pad), "Original", fill=(40, 40, 40), font=font)
    draw.text((pad * 2 + size + size // 2 - 35, pad), "Generated", fill=(40, 40, 40), font=font)

    for i, (orig, gen) in enumerate(zip(originals, generated_list)):
        y = label_h + pad + i * row_h
        canvas.paste(to_pil(orig).resize((size, size), Image.LANCZOS), (pad, y))
        canvas.paste(to_pil(gen).resize((size, size), Image.LANCZOS), (pad * 2 + size, y))

    return canvas


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-image SAE-conditioned inference")
    p.add_argument("--image", type=str, nargs="+", required=True, help="Input image path(s)")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Checkpoint path (ep-XXXX.pt) or 'latest' to auto-pick from results-dir")
    p.add_argument("--config", type=str,
                   default="configs/sae_rae/ImageNet256/DiTDH-XL_DINOv2-B_SAECLS.yaml")
    p.add_argument("--results-dir", type=str, default="ckpts/ft_sae_cls_batch_topk_ep50",
                   help="Used only when --ckpt=latest")
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--cfg-t-min", type=float, default=0.0)
    p.add_argument("--cfg-t-max", type=float, default=1.0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--top-k-show", type=int, default=40, help="Number of top SAE features to display")
    p.add_argument("--topk-crop", type=int, default=0,
                   help="Keep only top-k SAE activations as condition (0 = disabled)")
    p.add_argument("--sae-ckpt", type=str, default="",
                   help="Override SAE checkpoint path from config")
    p.add_argument("--out", type=str, default="infer_out", help="Output path prefix (no extension)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = Path(__file__).resolve().parent.parent
    add_sys_path(project_root / "vendor" / "rae_src")
    add_sys_path(project_root / "src")

    from stage1 import RAE
    from stage2.transport import create_transport, Sampler
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import center_crop_arr, parse_configs
    from sae_rae.conditioning import DinoClsSaeConditioner

    # ---- config ----
    full_cfg = OmegaConf.load(args.config)
    (rae_config, model_config, transport_config, _, _, _, training_config, _) = parse_configs(full_cfg)
    training_cfg = dict(OmegaConf.to_container(training_config, resolve=True)) if training_config else {}
    misc_cfg = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True))
    sae_cond_cfg = full_cfg.get("sae_condition")
    if sae_cond_cfg is None:
        raise ValueError("Config must include `sae_condition` block.")

    null_label = int(training_cfg.get("null_label", misc_cfg.get("num_classes", 1000)))
    latent_size = tuple(int(v) for v in misc_cfg.get("latent_size", [768, 16, 16]))

    # ---- resolve checkpoint ----
    if args.ckpt.strip().lower() == "latest":
        ckpts = _collect_epoch_checkpoints(Path(args.results_dir))
        if not ckpts:
            raise FileNotFoundError(f"No ep-*.pt found in {args.results_dir}")
        ckpt_path = ckpts[-1][1]
        print(f"[info] auto-selected checkpoint: {ckpt_path}")
    else:
        ckpt_path = Path(args.ckpt)

    # ---- RAE ----
    print("[info] loading RAE ...", flush=True)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval().requires_grad_(False)

    # ---- conditioner ----
    print("[info] loading conditioner (DINOv2 + SAE) ...", flush=True)
    s1_params = rae_config.get("params", {})
    conditioner = DinoClsSaeConditioner(
        encoder_config_path=str(sae_cond_cfg.get("encoder_config_path", s1_params.get("encoder_config_path"))),
        dinov2_path=str(sae_cond_cfg.get("dinov2_path", s1_params.get("encoder_params", {}).get("dinov2_path"))),
        encoder_input_size=int(sae_cond_cfg.get("encoder_input_size", s1_params.get("encoder_input_size", 224))),
        sae_ckpt_path=args.sae_ckpt if args.sae_ckpt else str(sae_cond_cfg.get("sae_ckpt")),
        sae_src_path=str(sae_cond_cfg.get("sae_src_path", str(project_root / "src"))),
    ).to(device)
    conditioner.eval().requires_grad_(False)

    # ---- DiT model ----
    print("[info] loading DiT model ...", flush=True)
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
        raise RuntimeError("Checkpoint has neither 'ema' nor 'model'.")
    model.eval().requires_grad_(False)

    # ---- transport / sampler ----
    transport_params = dict(transport_config.get("params", {}))
    shift_dim = int(full_cfg.get("misc", {}).get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(full_cfg.get("misc", {}).get("time_dist_shift_base", 4096))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=(shift_dim / shift_base) ** 0.5)
    sample_fn = Sampler(transport).sample_ode(sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3)

    # ---- load & preprocess images ----
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.ToTensor(),
    ])
    img_tensors = []
    for path in args.image:
        print(f"[info] loading image: {path}", flush=True)
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
        img_tensors.append(transform(pil_img))
    img_batch = torch.stack(img_tensors).to(device)  # (N, 3, H, W)
    N = img_batch.shape[0]

    # ---- extract SAE latents ----
    print("[info] extracting SAE latents ...", flush=True)
    with torch.no_grad():
        cond = conditioner(img_batch)  # (N, cond_dim)

    if args.topk_crop > 0:
        topk_vals, topk_idx = torch.topk(cond.abs(), args.topk_crop, dim=1)
        mask = torch.zeros_like(cond)
        mask.scatter_(1, topk_idx, 1.0)
        cond = cond * mask
        print(f"[info] topk-crop applied: keeping top-{args.topk_crop} activations per image")

    for i, path in enumerate(args.image):
        latent_vec = cond[i]
        nonzero = (latent_vec != 0).sum().item()
        total = latent_vec.numel()
        print(f"[image {i}] {Path(path).name}  active={nonzero}/{total}  sparsity={100*(total-nonzero)/total:.1f}%")
        top_k_print = min(10, nonzero)
        if top_k_print > 0:
            vals, idxs = torch.topk(latent_vec, top_k_print)
            for v, idx in zip(vals.tolist(), idxs.tolist()):
                print(f"  feature {idx:5d} = {v:.4f}")

    # ---- generate ----
    print("[info] generating images (ODE sampler) ...", flush=True)
    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    y_null = torch.full((N,), null_label, device=device, dtype=torch.long)

    with torch.no_grad():
        if args.cfg_scale > 1.0:
            z0 = torch.randn(N, *latent_size, device=device)
            z = torch.cat([z0, z0], dim=0)            # (2N,)
            y_cfg = torch.cat([y_null, y_null], dim=0)
            cond_cfg = torch.cat([cond, torch.zeros_like(cond)], dim=0)

            def model_fn(x, t, y, cfg_scale, cfg_interval=(0.0, 1.0), cond=None):
                model_out = model.forward(x, t, y, cond=cond)
                eps, rest = model_out[:, :model.in_channels], model_out[:, model.in_channels:]
                cond_eps, uncond_eps = eps[:N], eps[N:]
                guided = torch.where(
                    ((t[:N] >= cfg_interval[0]) & (t[:N] <= cfg_interval[1])).view(-1, *[1] * (len(cond_eps.shape) - 1)),
                    uncond_eps + cfg_scale * (cond_eps - uncond_eps),
                    cond_eps,
                )
                return torch.cat([torch.cat([guided, guided], dim=0), rest], dim=1)

            with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                zhat = sample_fn(
                    z, model_fn, y=y_cfg, cond=cond_cfg,
                    cfg_scale=args.cfg_scale,
                    cfg_interval=(args.cfg_t_min, args.cfg_t_max),
                )[-1][:N]
                generated = rae.decode(zhat).clamp(0, 1)
        else:
            z0 = torch.randn(N, *latent_size, device=device)
            with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                zhat = sample_fn(z0, model.forward, y=y_null, cond=cond)[-1]
                generated = rae.decode(zhat).clamp(0, 1)

    # ---- save outputs ----
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    comparison_path = Path(str(out_prefix) + "_comparison.png")
    comparison = make_comparison(
        [img_batch[i] for i in range(N)],
        [generated[i] for i in range(N)],
        size=args.image_size,
    )
    comparison.save(comparison_path)
    print(f"[saved] {comparison_path}")

    for i in range(N):
        suffix = f"_latent_{i}.png" if N > 1 else "_latent.png"
        latent_path = Path(str(out_prefix) + suffix)
        latent_chart = make_latent_chart(cond[i], top_k=args.top_k_show)
        latent_chart.save(latent_path)
        print(f"[saved] {latent_path}")


if __name__ == "__main__":
    main()

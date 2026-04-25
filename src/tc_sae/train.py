#!/usr/bin/env python3
"""SAE + FactorVAE-style TC regularization (from-scratch 학습).

학습 구조 (per batch):
  Step A. Discriminator update  (SAE frozen)
    - real joint z       → label 1
    - shuffled marginals → label 0
    - disc_steps 번 반복

  Step B. SAE update  (discriminator frozen)
    - L_total = L_nmse + λ_L1 * L_L1 + γ_TC * E[logit(D(z))]
    - γ_TC는 tc_warmup_epochs 동안 0 → tc_weight 선형 증가

실행:
    python src/tc_sae/train.py \\
        --config configs/tc_sae/ffhq256_sae_tc_preact_v1.yaml \\
        [--wandb]

wandb 로깅:
  train/nmse        — SAE reconstruction loss
  train/l1          — L1 sparsity loss
  train/tc_loss     — TC penalty (E[logit(D(z))])
  train/disc_loss   — discriminator BCE
  train/tc_weight   — 현재 γ_TC 값
  eval/val_nmse     — validation NMSE
  eval/dead_ratio   — dead feature 비율
"""
from __future__ import annotations  # noqa: F401 – Python 3.9 union-type compat

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# HuggingFace 로딩 progress bar 억제 (tqdm 전체 비활성화 없이)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, Dinov2WithRegistersModel

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import wandb as _wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── path setup ────────────────────────────────────────────────────────────────

def _add_path(p: str) -> None:
    if p not in sys.path:
        sys.path.insert(0, p)


def _resolve_path(path_str: str, repo_root: Path, config_path: Path | None = None) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    candidates = []
    if config_path is not None:
        candidates.append((config_path.parent / p).resolve())
    candidates.append((repo_root / p).resolve())

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── DINOv2 CLS extractor ──────────────────────────────────────────────────────

class DinoClsExtractor(nn.Module):
    def __init__(self, encoder_config_path: str, dinov2_path: str,
                 encoder_input_size: int) -> None:
        super().__init__()
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(proc.image_std ).view(1,3,1,1), persistent=False)
        self.encoder_input_size = int(encoder_input_size)
        self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path)
        self.encoder.requires_grad_(False)
        self.hidden_size = int(self.encoder.config.hidden_size)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, _, h, w = images.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            images = F.interpolate(images,
                                   size=(self.encoder_input_size, self.encoder_input_size),
                                   mode="bicubic", align_corners=False)
        x = (images - self.mean) / self.std
        return self.encoder(x).last_hidden_state[:, 0, :]


def _build_extractor(rae_config_path: str, device: torch.device) -> DinoClsExtractor:
    cfg = OmegaConf.load(rae_config_path)
    p = cfg.stage_1.params
    enc_cfg   = str(p.get("encoder_config_path", "facebook/dinov2-with-registers-base"))
    enc_size  = int(p.get("encoder_input_size", 224))
    dino_path = str(p.get("encoder_params", {}).get("dinov2_path", enc_cfg))
    ext = DinoClsExtractor(enc_cfg, dino_path, enc_size).to(device)
    ext.eval()
    return ext


# ── data ──────────────────────────────────────────────────────────────────────

from torch.utils.data import Dataset

class CachedFeatureDataset(Dataset):
    """Pre-extracted DINOv2 CLS tokens stored as numpy memmap (.bin)."""

    def __init__(self, features_path: str):
        p = Path(features_path)
        shape_path = p.parent / (p.stem + "_shape.npy")
        shape = tuple(int(v) for v in np.load(str(shape_path)))
        self.features = np.memmap(features_path, dtype=np.float32, mode="r", shape=shape)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.features[idx].copy()), 0


def _build_loader(data_path: str, image_size: int, batch_size: int,
                  num_workers: int, shuffle: bool = True) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    ds = ImageFolder(data_path, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=shuffle, persistent_workers=(num_workers > 0),
                      prefetch_factor=2 if num_workers > 0 else None)


# ── TC warmup schedule ────────────────────────────────────────────────────────

def get_tc_weight(epoch: int, tc_weight: float, tc_warmup_epochs: int) -> float:
    """epoch 1→0, epoch tc_warmup_epochs→tc_weight 선형 증가, 이후 tc_weight 유지.

    epoch 1: 1/tc_warmup_epochs
    epoch N (N>=tc_warmup_epochs): tc_weight
    """
    if tc_warmup_epochs <= 0:
        return tc_weight
    return tc_weight * min(1.0, epoch / tc_warmup_epochs)


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(
    sae, extractor, loader, device, max_batches, nmse_fn,
    co_act_sample: int = 512,
    disc=None,
) -> dict:
    """Evaluate SAE reconstruction, sparsity, and co-activation.

    co_act_sample: co-activation matrix 계산에 쓸 feature 샘플 수.
                   전체 6144×6144은 너무 크므로 random 512개 feature만 사용.
    disc:          discriminator 모델. 주어지면 disc_real_acc 계산.
    """
    sae.eval()
    if disc is not None:
        disc.eval()
    nmse_sum  = 0.0
    n_steps   = 0
    n_latents = sae.encoder.weight.shape[0]
    act_count = torch.zeros(n_latents, dtype=torch.long)

    # co-activation 계산용 accumulator
    # active 벡터를 전부 모으면 메모리 부담 → 배치별로 outer product 누적
    # C_sum[i,j] += Σ_b active[b,i] * active[b,j]  (sample-wise)
    # 전체 feature pair 대신 고정 sample index만 사용
    rng = torch.Generator()
    rng.manual_seed(42)
    sample_idx = torch.randperm(n_latents, generator=rng)[:co_act_sample]  # [S]
    co_sum  = torch.zeros(co_act_sample, co_act_sample, dtype=torch.float32)
    n_total = 0  # total samples seen

    # mean_abs_pearson 계산용: 연속 activation 값 누적 (sampled features)
    latents_buf: list[torch.Tensor] = []

    # disc_real_acc 계산용
    disc_acc_sum   = 0.0
    disc_acc_steps = 0

    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        if extractor is None:
            cls = batch[0].to(device, non_blocking=True)
        else:
            imgs = batch[0].to(device, non_blocking=True)
            cls = extractor(imgs)
        cls_norm, info = sae.preprocess(cls)
        z_pre        = sae.encode_pre_act(cls_norm)
        latents      = sae.activation(z_pre)
        recons       = sae.decode(latents, info)
        nmse_sum += float(nmse_fn(recons, cls).item())
        n_steps  += 1

        active = (latents > 0).float()                    # [B, n_latents]
        act_count += active.long().sum(dim=0).cpu()

        # sampled co-activation (binary)
        a_sub = active[:, sample_idx].cpu()               # [B, S]
        co_sum += a_sub.T @ a_sub                         # [S, S]
        n_total += active.shape[0]

        # mean_abs_pearson용: 연속값 샘플링
        latents_buf.append(latents[:, sample_idx].cpu())  # [B, S]

        # disc_real_acc: joint(real) vs per-column shuffled(fake) 판별 정확도
        if disc is not None:
            # per-column shuffle: pre-activation continuous latent 전용
            _B, _D  = z_pre.shape
            _perm   = torch.argsort(torch.rand(_B, _D, device=device), dim=0)
            _col    = torch.arange(_D, device=device).unsqueeze(0).expand(_B, _D)
            z_shuf  = z_pre[_perm, _col]
            logit_r = disc(z_pre)                          # joint → label 1
            logit_f = disc(z_shuf)                         # permuted → label 0
            acc = float(
                ((logit_r > 0).float().mean() + (logit_f < 0).float().mean()) / 2
            )
            disc_acc_sum   += acc
            disc_acc_steps += 1

    dead_ratio = float((act_count == 0).float().mean())
    avg_active = float(act_count.float().sum() / max(n_total, 1))  # 샘플당 평균 active feature 수

    # co-activation rate matrix: C[i,j] = P(feat_i active AND feat_j active)
    co_rate = co_sum / max(n_total, 1)                    # [S, S]

    # off-diagonal mean (대각은 P(feat_i active)² 으로 self-correlation)
    mask = ~torch.eye(co_act_sample, dtype=torch.bool)
    co_act_mean = float(co_rate[mask].mean().item())

    # independence baseline: features가 완전히 독립이면
    # P(i∩j) = P(i)*P(j) → off-diag ≈ (k/n_latents)²
    k_avg = float(act_count.float().mean() / max(n_total, 1))
    co_act_baseline = k_avg ** 2

    # excess co-activation: baseline 대비 얼마나 더 co-activate하는가
    # 이 값이 epoch에 걸쳐 줄어들면 TC가 실제로 작동하고 있는 것
    co_act_excess = co_act_mean - co_act_baseline

    # mean_abs_pearson: 연속 activation 값 기반 feature 간 선형 상관관계 평균
    # TC가 잘 작동하면 이 값이 줄어야 함 (co_act보다 훨씬 민감한 지표)
    z_all = torch.cat(latents_buf, dim=0).float()         # [N, S]
    del latents_buf
    z_c   = z_all - z_all.mean(0)
    del z_all
    z_c   = z_c / z_c.std(0).clamp(min=1e-6)
    corr  = (z_c.T @ z_c) / max(z_c.shape[0] - 1, 1)    # [S, S]
    del z_c
    mean_abs_pearson = float(corr.abs()[mask].mean().item())
    del corr

    # act_gini: feature 활성화 빈도의 불균형 (0=완전균일, 1=완전집중)
    # TC가 feature를 고르게 사용하게 만들면 낮아져야 함
    freq     = act_count.float() / max(n_total, 1)        # [n_latents]
    sorted_f = freq.sort().values
    n_f      = len(freq)
    gini     = float(
        1.0 - 2.0 * sorted_f.cumsum(0)[:-1].sum()
        / (n_f * sorted_f.sum().clamp(min=1e-9))
    )

    # disc_real_acc: 0.5 = discriminator가 구분 못함 (features 독립) ← TC 목표
    #               1.0 = 완벽 판별 (features 여전히 종속)
    disc_real_acc = (disc_acc_sum / max(disc_acc_steps, 1)
                     if disc is not None else None)

    result = {
        "val_nmse":          nmse_sum / max(n_steps, 1),
        "dead_ratio":        dead_ratio,
        "avg_active":        avg_active,
        "co_act_mean":       co_act_mean,        # 실제 co-activation rate (binary)
        "co_act_baseline":   co_act_baseline,    # 독립 가정 baseline
        "co_act_excess":     co_act_excess,      # binary excess
        "mean_abs_pearson":  mean_abs_pearson,   # 연속값 기반 상관관계 (핵심)
        "act_gini":          gini,               # feature 활성화 불균형
    }
    if disc_real_acc is not None:
        result["disc_real_acc"] = disc_real_acc  # 0.5 목표
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str, required=True)
    parser.add_argument("--resume",         type=str, default=None,
                        help="checkpoint path to resume from (omit to auto-detect save_dir/sae_tc_latest.pt)")
    parser.add_argument("--no-resume",      action="store_true",
                        help="disable auto-resume even if sae_tc_latest.pt exists")
    parser.add_argument("--wandb",          action="store_true")
    parser.add_argument("--wandb-project",  type=str, default="SAE-DINO-TC")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-run-id",   type=str, default=None,
                        help="wandb run id to resume (use with --resume)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config).resolve()
    cfg      = OmegaConf.load(args.config)
    exp_cfg  = cfg.experiment
    path_cfg = cfg.paths
    data_cfg = cfg.data
    sae_cfg  = cfg.sae
    tc_cfg   = cfg.get("tc", {})
    eval_cfg = cfg.get("eval", {})

    # 프로젝트 내부 src만 import
    tc_src = str(repo_root / "src")
    _add_path(tc_src)

    from sae_local.model import Autoencoder, TopK
    from sae_local.loss  import autoencoder_loss, normalized_mean_squared_error
    from tc_sae.discriminator import Discriminator, shuffle_latents
    from tc_sae.discriminator import discriminator_loss, tc_penalty

    device = torch.device(str(exp_cfg.device)
                          if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    _set_seed(int(exp_cfg.seed))

    save_dir = Path(_resolve_path(str(exp_cfg.save_dir), repo_root, config_path))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── SAE ──────────────────────────────────────────────────────────────
    cache_cfg = cfg.get("features_cache", None)
    use_feature_cache = cache_cfg is not None and bool(cache_cfg.get("enabled", True))

    if use_feature_cache:
        _feat_p = Path(str(cache_cfg.train_features))
        _shape_file = _feat_p.parent / (_feat_p.stem + "_shape.npy")
        d_model = int(np.load(str(_shape_file))[1])
        extractor = None
        print(f"[info] using cached features: {_feat_p} (d_model={d_model})", flush=True)
    else:
        rae_config_path = _resolve_path(str(path_cfg.rae_config), repo_root, config_path)
        extractor = _build_extractor(rae_config_path, device)
        d_model   = int(extractor.hidden_size)

    k = int(sae_cfg.k)
    activation = TopK(k=k) if k > 0 else nn.ReLU()
    sae = Autoencoder(
        n_latents  = int(sae_cfg.n_latents),
        n_inputs   = d_model,
        activation = activation,
        tied       = False,
        normalize  = False,
    ).to(device)
    print(f"[info] SAE: d_model={d_model} n_latents={sae_cfg.n_latents} k={k}")

    # ── discriminator ─────────────────────────────────────────────────────
    tc_enabled    = bool(tc_cfg.get("enabled", True))
    tc_weight_max = float(tc_cfg.get("tc_weight", 0.05))
    tc_warmup     = int(tc_cfg.get("tc_warmup_epochs", 10))
    disc_lr       = float(tc_cfg.get("disc_lr", 1e-4))
    disc_hidden   = int(tc_cfg.get("disc_hidden", 512))
    disc_steps       = int(tc_cfg.get("disc_steps", 1))
    disc_update_freq = int(tc_cfg.get("disc_update_freq", 1))  # SAE K step마다 disc 1회 업데이트
    sae_grad_clip    = float(tc_cfg.get("sae_grad_clip", 1.0))

    disc = None
    disc_optimizer = None
    if tc_enabled:
        disc = Discriminator(n_lat=int(sae_cfg.n_latents), hidden=disc_hidden).to(device)
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=disc_lr)
        print(f"[info] Discriminator: input={sae_cfg.n_latents} hidden={disc_hidden} "
              f"tc_weight={tc_weight_max} warmup={tc_warmup}ep "
              f"disc_update_freq={disc_update_freq} "
              f"sae_grad_clip={sae_grad_clip}")

    # ── data loaders ──────────────────────────────────────────────────────
    if use_feature_cache:
        train_loader = DataLoader(
            CachedFeatureDataset(str(cache_cfg.train_features)),
            batch_size=int(data_cfg.batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
    else:
        train_loader = _build_loader(str(path_cfg.data_path), int(data_cfg.image_size),
                                     int(data_cfg.batch_size), int(data_cfg.num_workers))

    do_eval    = bool(eval_cfg.get("enabled", False))
    val_loader = None
    if do_eval:
        if use_feature_cache and cache_cfg.get("eval_features", None):
            val_loader = DataLoader(
                CachedFeatureDataset(str(cache_cfg.eval_features)),
                batch_size=int(data_cfg.batch_size),
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )
        else:
            val_loader = _build_loader(str(eval_cfg.data_path), int(data_cfg.image_size),
                                       int(data_cfg.batch_size), int(data_cfg.num_workers),
                                       shuffle=False)

    # ── SAE optimizer ─────────────────────────────────────────────────────
    sae_optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=float(sae_cfg.lr),
        weight_decay=float(sae_cfg.weight_decay),
    )

    # AMP: fp16일 때만 scaler 필요 (bf16은 불필요)
    use_amp   = bool(exp_cfg.get("use_amp", True)) and (device.type == "cuda")
    amp_dtype = (torch.float16 if str(exp_cfg.get("amp_dtype","bf16")).lower() == "fp16"
                 else torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(amp_dtype == torch.float16 and use_amp))

    start_epoch   = 1
    best_val_nmse = float("inf")

    # ── resume ────────────────────────────────────────────────────────────
    resume_path = args.resume
    if resume_path is None and not args.no_resume:
        auto = save_dir / "sae_tc_latest.pt"
        if auto.exists():
            resume_path = str(auto)

    if resume_path:
        print(f"[resume] loading {resume_path}")
        ckpt_r = torch.load(resume_path, map_location=device)
        sae.load_state_dict(ckpt_r["sae_state_dict"], strict=False)
        sae_optimizer.load_state_dict(ckpt_r["optimizer_state_dict"])
        if disc is not None and "disc_state_dict" in ckpt_r:
            disc.load_state_dict(ckpt_r["disc_state_dict"])
            disc_optimizer.load_state_dict(ckpt_r["disc_optimizer_state"])
        start_epoch   = int(ckpt_r["epoch"]) + 1
        best_val_nmse = float(ckpt_r.get("val_nmse", float("inf")))
        print(f"[resume] → start epoch {start_epoch}  best_val_nmse={best_val_nmse:.6f}")
        del ckpt_r

    # ── wandb ─────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or Path(args.config).stem
        wandb_kwargs: dict = dict(
            project=args.wandb_project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        if args.wandb_run_id:
            wandb_kwargs["id"]     = args.wandb_run_id
            wandb_kwargs["resume"] = "must"
        wandb_run = _wandb.init(**wandb_kwargs)

    log_every   = int(exp_cfg.get("log_interval_steps", 100))
    save_every  = int(exp_cfg.save_every)
    eval_every  = int(eval_cfg.get("eval_every", 1))
    max_batches = int(eval_cfg.get("max_batches", 0))
    epochs      = int(sae_cfg.epochs)
    l1_weight   = float(sae_cfg.l1_weight)

    print(f"[info] train_batches={len(train_loader)}  epochs={epochs}")

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        sae.train()
        if disc is not None:
            disc.train()

        cur_tc_w = get_tc_weight(epoch, tc_weight_max, tc_warmup) if tc_enabled else 0.0

        t0 = time.perf_counter()
        nmse_sum = l1_sum = tc_sum = disc_sum = disc_acc_sum = 0.0
        n_steps  = 0

        for step, batch in enumerate(train_loader, start=1):
            if use_feature_cache:
                cls = batch[0].to(device, non_blocking=True)
            else:
                imgs = batch[0].to(device, non_blocking=True)
                with torch.inference_mode():
                    cls = extractor(imgs)

            amp_ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                       if use_amp else nullcontext())

            # ── Step A: Discriminator update ──────────────────────────────
            # disc_update_freq: SAE K step마다 disc 1회 업데이트
            if tc_enabled and disc is not None and disc_optimizer is not None:
                with torch.no_grad():
                    cls_norm, _ = sae.preprocess(cls)
                    z_pre_detach = sae.encode_pre_act(cls_norm)
                    z_shuf = shuffle_latents(z_pre_detach)

                if step % disc_update_freq == 0:
                    for _ in range(disc_steps):
                        d_loss = discriminator_loss(disc, z_pre_detach, z_shuf)
                        disc_optimizer.zero_grad(set_to_none=True)
                        d_loss.backward()
                        disc_optimizer.step()
                    disc_sum += float(d_loss.item())
                else:
                    with torch.no_grad():
                        logit_r_tmp = disc(z_pre_detach)
                        logit_f_tmp = disc(z_shuf)
                    disc_sum += float(0.5 * (
                        F.binary_cross_entropy_with_logits(logit_r_tmp, torch.ones_like(logit_r_tmp))
                        + F.binary_cross_entropy_with_logits(logit_f_tmp, torch.zeros_like(logit_f_tmp))
                    ).item())

                # disc_acc: 업데이트 후 정확도 측정 (post-update)
                with torch.no_grad():
                    logit_r = disc(z_pre_detach)
                    logit_f = disc(z_shuf)
                    cur_acc = float(
                        ((logit_r > 0).float().mean() + (logit_f < 0).float().mean()) / 2
                    )
                disc_acc_sum += cur_acc

            # ── Step B: SAE update ────────────────────────────────────────
            if disc is not None:
                disc.requires_grad_(False)

            with amp_ctx:
                cls_norm, info = sae.preprocess(cls)
                z_pre   = sae.encode_pre_act(cls_norm)
                latents = sae.activation(z_pre)
                recons  = sae.decode(latents, info)

                recon_loss = normalized_mean_squared_error(recons, cls)
                l1_loss    = (latents.abs().sum(dim=1) / cls.norm(dim=1)).mean()
                tc_loss    = (tc_penalty(disc, z_pre)
                              if (tc_enabled and disc is not None and cur_tc_w > 0.0)
                              else torch.tensor(0.0, device=device))

                total_loss = recon_loss + l1_weight * l1_loss + cur_tc_w * tc_loss

            sae_optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            # SAE gradient clipping: TC gradient가 폭발적으로 커지는 것을 방지
            if sae_grad_clip > 0:
                scaler.unscale_(sae_optimizer)
                torch.nn.utils.clip_grad_norm_(sae.parameters(), sae_grad_clip)
            scaler.step(sae_optimizer)
            scaler.update()

            if disc is not None:
                disc.requires_grad_(True)

            nmse_sum += float(recon_loss.item())
            l1_sum   += float(l1_loss.item())
            tc_sum   += float(tc_loss.item())
            n_steps  += 1

            if log_every > 0 and step % log_every == 0:
                elapsed  = time.perf_counter() - t0
                it_per_s = step / max(elapsed, 1e-6)
                eta_min  = (len(train_loader) - step) / max(it_per_s, 1e-6) / 60
                avg_n    = nmse_sum     / n_steps
                avg_l1   = l1_sum       / n_steps
                avg_tc   = tc_sum       / n_steps
                avg_d    = disc_sum     / max(n_steps, 1)
                avg_dacc = disc_acc_sum / max(n_steps, 1)
                print(
                    f"[epoch {epoch:03d} step {step:04d}/{len(train_loader):04d}] "
                    f"nmse={avg_n:.5f}  l1={avg_l1:.5f}  "
                    f"tc={avg_tc:.5f}  disc={avg_d:.5f}  disc_acc={avg_dacc:.4f}  "
                    f"γ={cur_tc_w:.4f}  it/s={it_per_s:.2f}  eta={eta_min:.1f}m",
                    flush=True,
                )
                if wandb_run:
                    gstep = (epoch - 1) * len(train_loader) + step
                    wandb_run.log({
                        "train/nmse":      avg_n,
                        "train/l1":        avg_l1,
                        "train/tc_loss":   avg_tc,
                        "train/disc_loss": avg_d,
                        "train/disc_acc":  avg_dacc,
                        "train/tc_weight": cur_tc_w,
                    }, step=gstep)

        avg_n    = nmse_sum     / max(n_steps, 1)
        avg_l1   = l1_sum       / max(n_steps, 1)
        avg_tc   = tc_sum       / max(n_steps, 1)
        avg_d    = disc_sum     / max(n_steps, 1)
        avg_dacc = disc_acc_sum / max(n_steps, 1)
        print(f"[epoch {epoch:03d}] "
              f"nmse={avg_n:.6f}  l1={avg_l1:.6f}  "
              f"tc={avg_tc:.6f}  disc={avg_d:.6f}  disc_acc={avg_dacc:.4f}  γ={cur_tc_w:.4f}")

        if wandb_run:
            wandb_run.log({
                "epoch/nmse":      avg_n,
                "epoch/l1":        avg_l1,
                "epoch/tc_loss":   avg_tc,
                "epoch/disc_loss": avg_d,
                "epoch/disc_acc":  avg_dacc,
                "epoch/tc_weight": cur_tc_w,
                "epoch":           epoch,
            }, step=epoch * len(train_loader))

        # ── save ──────────────────────────────────────────────────────────
        ckpt = {
            "epoch":               epoch,
            "sae_state_dict":      sae.state_dict(),
            "optimizer_state_dict":sae_optimizer.state_dict(),
            "n_inputs":            d_model,
            "n_latents":           int(sae_cfg.n_latents),
            "k":                   k,
            "l1_weight":           l1_weight,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if disc is not None:
            ckpt["disc_state_dict"]      = disc.state_dict()
            ckpt["disc_optimizer_state"] = disc_optimizer.state_dict()
        # 매 epoch: latest.pt 덮어쓰기 (서버 재시작 대비)
        torch.save(ckpt, save_dir / "sae_tc_latest.pt")
        # save_every epoch: 번호 붙은 체크포인트도 저장
        if epoch % save_every == 0:
            ckpt_path = save_dir / f"sae_tc_epoch{epoch:03d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"[save] {ckpt_path}")

        # ── eval ──────────────────────────────────────────────────────────
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if do_eval and val_loader and (epoch % eval_every == 0):
            metrics = evaluate(
                sae=sae, extractor=extractor, loader=val_loader,
                device=device, max_batches=max_batches,
                nmse_fn=normalized_mean_squared_error,
                disc=disc,
            )
            metrics["tc_weight"] = cur_tc_w
            disc_acc_str = (f"  disc_acc={metrics['disc_real_acc']:.4f}"
                            if "disc_real_acc" in metrics else "")
            print(
                f"[eval {epoch:03d}] "
                f"val_nmse={metrics['val_nmse']:.6f}  "
                f"dead={metrics['dead_ratio']:.4f}  "
                f"pearson={metrics['mean_abs_pearson']:.6f}  "
                f"gini={metrics['act_gini']:.4f}"
                f"{disc_acc_str}  "
                f"co_excess={metrics['co_act_excess']:.6f}",
                flush=True,
            )
            metric_path = save_dir / f"epoch{epoch:03d}_metrics.json"
            with open(metric_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # ── best checkpoint ───────────────────────────────────────────
            if metrics["val_nmse"] < best_val_nmse:
                best_val_nmse = metrics["val_nmse"]
                best_path = save_dir / "sae_tc_best.pt"
                best_ckpt = {
                    "epoch":               epoch,
                    "val_nmse":            best_val_nmse,
                    "sae_state_dict":      sae.state_dict(),
                    "optimizer_state_dict":sae_optimizer.state_dict(),
                    "n_inputs":            d_model,
                    "n_latents":           int(sae_cfg.n_latents),
                    "k":                   k,
                    "l1_weight":           l1_weight,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }
                if disc is not None:
                    best_ckpt["disc_state_dict"]      = disc.state_dict()
                    best_ckpt["disc_optimizer_state"] = disc_optimizer.state_dict()
                torch.save(best_ckpt, best_path)
                print(f"[best] val_nmse={best_val_nmse:.6f}  → {best_path}")

            if wandb_run:
                log_dict = {
                    "eval/val_nmse":         metrics["val_nmse"],
                    "eval/dead_ratio":       metrics["dead_ratio"],
                    "eval/avg_active":       metrics["avg_active"],
                    "eval/mean_abs_pearson": metrics["mean_abs_pearson"],  # 핵심 disentangle 지표
                    "eval/act_gini":         metrics["act_gini"],
                    "eval/co_act_mean":      metrics["co_act_mean"],
                    "eval/co_act_baseline":  metrics["co_act_baseline"],
                    "eval/co_act_excess":    metrics["co_act_excess"],
                }
                if "disc_real_acc" in metrics:
                    log_dict["eval/disc_real_acc"] = metrics["disc_real_acc"]
                wandb_run.log(log_dict, step=epoch * len(train_loader))

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()

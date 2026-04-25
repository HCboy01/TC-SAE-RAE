"""Discriminator for FactorVAE-style TC regularization on SAE latents.

D(z) ∈ R  distinguishes:
  - real joint:         z  ~ q(z)          (SAE pre-activation latent)
  - shuffled marginals: z̃ ~ Π_j q(z_j)   (차원별 독립 shuffle)

At optimum:  logit(D(z)) ≈ log q(z) / Π_j q(z_j)
따라서 TC(z) ≈ E_q[logit(D(z))]

SAE 업데이트에서 E_q[logit(D(z))]를 최소화하면
discriminator가 joint를 shuffled로부터 구분하지 못하게 됨
→ feature 간 spurious dependence 감소.

Discriminator 입력: TopK 통과 전 pre-activation latent (continuous dense)
  - count shortcut 없음: sparse하지 않으므로 활성 뉴런 수라는 trivial cue 자체가 없음
  - per-column shuffle이 이론적으로 올바른 ∏_j q(z_j) 근사
  - batch dependence 없음
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """Small MLP discriminator for SAE latent space.

    Args:
        n_lat:  total latent dim (e.g. 6144)
        hidden: first hidden layer width
    """

    def __init__(self, n_lat: int, hidden: int = 512) -> None:
        super().__init__()
        if hidden == 0:
            # 선형 판별기 (sparse 벡터에서 학습 불가 → 사용 비권장)
            self.net = nn.Sequential(nn.Linear(n_lat, 1))
        else:
            # 단일 hidden layer: n_lat → hidden → 1
            # 2-layer MLP 대비 표현력 제한 → disc_acc 상한이 낮아짐
            self.net = nn.Sequential(
                nn.Linear(n_lat, hidden),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden, 1),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, n_lat]  pre-activation latent (continuous, TopK 통과 전)
        Returns:
            logits: [B]  raw logits (no sigmoid)
                    high  →  looks like real joint
                    low   →  looks like shuffled marginals
        """
        return self.net(z).squeeze(-1)


def shuffle_latents(z: torch.Tensor) -> torch.Tensor:
    """Per-feature (per-column) shuffle for continuous pre-activation latents.

    각 feature dimension j를 배치 내에서 독립적으로 permute.
    이론적으로 ∏_j p(z_j) — product of marginals — 에서 샘플한 것과 동일.

    pre-activation (continuous dense) latent 전용.
    count shortcut 문제가 없으므로 특별한 처리 불필요.

    Args:
        z: [B, D]  pre-activation latent (continuous)
    Returns:
        z_shuffled: [B, D]  per-feature shuffled latent
    """
    B, D = z.shape
    perm = torch.argsort(torch.rand(B, D, device=z.device), dim=0)  # [B, D]
    col  = torch.arange(D, device=z.device).unsqueeze(0).expand(B, D)
    return z[perm, col]


def discriminator_loss(
    disc: Discriminator,
    z_real: torch.Tensor,
    z_shuffled: torch.Tensor,
) -> torch.Tensor:
    """Discriminator BCE loss.

    Real joint    → label 1
    Shuffled marg → label 0

    Args:
        disc:       Discriminator
        z_real:     [B, n_lat]  real joint latent  (반드시 detach)
        z_shuffled: [B, n_lat]  shuffled latent    (반드시 detach)
    Returns:
        scalar loss
    """
    logit_r = disc(z_real)
    logit_f = disc(z_shuffled)
    loss = (
        F.binary_cross_entropy_with_logits(logit_r, torch.ones_like(logit_r))
        + F.binary_cross_entropy_with_logits(logit_f, torch.zeros_like(logit_f))
    )
    return loss * 0.5


def tc_penalty(
    disc: Discriminator,
    z_real: torch.Tensor,
) -> torch.Tensor:
    """SAE에 더하는 TC penalty.

    TC ≈ E_q[logit(D(z))]  (FactorVAE density-ratio trick)
    이 값을 최소화하면 D(z) → 0.5  →  joint가 marginals처럼 보임.

    Args:
        disc:    Discriminator  (SAE update 시 parameters는 frozen)
        z_real:  [B, n_lat]  SAE latent, gradient 흘러야 함
    Returns:
        scalar TC penalty
    """
    return disc(z_real).mean()

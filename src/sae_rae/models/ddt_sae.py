from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage2.models.DDT import DiTwDDTHead


class DiTwDDTHeadSAECond(DiTwDDTHead):
    """DiT^DH with optional global SAE condition injection.

    The base DiT weights can be loaded from pretraining checkpoints while keeping
    their initial behavior, because `cond_gate` is initialized to zeros.
    """

    def __init__(
        self,
        *args,
        cond_dim: int = 0,
        cond_dropout_prob: float = 0.1,
        cond_use_layernorm: bool = False,  # deprecated, ignored
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cond_dim = int(cond_dim)
        self.cond_dropout_prob = float(cond_dropout_prob)

        if self.cond_dim > 0:
            self.cond_proj = nn.Linear(self.cond_dim, self.encoder_hidden_size, bias=True)
            self.cond_gate = nn.Parameter(torch.zeros(self.encoder_hidden_size))
            nn.init.normal_(self.cond_proj.weight, std=0.02)
            nn.init.zeros_(self.cond_proj.bias)
        else:
            self.cond_proj = None
            self.cond_gate = None

    def _build_cond(self, cond: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if self.cond_proj is None or cond is None:
            return None
        if cond.ndim != 2:
            raise ValueError(f"Expected cond shape (B, D), got {tuple(cond.shape)}")
        if cond.shape[0] != batch_size:
            raise ValueError(f"cond batch ({cond.shape[0]}) must match input batch ({batch_size})")

        cond = self.cond_proj(cond)
        if self.training and self.cond_dropout_prob > 0:
            keep_prob = 1.0 - self.cond_dropout_prob
            keep_mask = (torch.rand(cond.shape[0], 1, device=cond.device) < keep_prob).to(cond.dtype)
            cond = cond * keep_mask

        cond = cond * self.cond_gate.unsqueeze(0)
        return cond

    def _align_cfg_cond(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor | None:
        if cond is None:
            return None
        if cond.shape[0] == x.shape[0]:
            half = cond[: cond.shape[0] // 2]
            return torch.cat([half, torch.zeros_like(half)], dim=0)
        if cond.shape[0] == x.shape[0] // 2:
            return torch.cat([cond, torch.zeros_like(cond)], dim=0)
        raise ValueError(
            f"Invalid cond batch size for CFG: cond={cond.shape[0]}, x={x.shape[0]}"
        )

    def forward(self, x, t, y=None, s=None, mask=None, cond: torch.Tensor | None = None):
        t = self.t_embedder(t)
        c = t

        cond_vec = self._build_cond(cond, batch_size=x.shape[0])
        if cond_vec is not None:
            c = c + cond_vec
        c = F.silu(c)

        if s is None:
            s = self.s_embedder(x)
            if self.use_pos_embed:
                s = s + self.pos_embed
            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
            t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = F.silu(t + s)

        s = self.s_projector(s)
        x = self.x_embedder(x)
        if self.use_pos_embed and self.x_pos_embed is not None:
            x = x + self.x_pos_embed
        for i in range(self.num_encoder_blocks, self.num_blocks):
            x = self.blocks[i](x, s, feat_rope=self.dec_feat_rope)
        x = self.final_layer(x, s)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(0, 1), cond: torch.Tensor | None = None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        cond_combined = self._align_cfg_cond(x, cond)

        model_out = self.forward(combined, t, y, cond=cond_combined)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        guid_t_min, guid_t_max = cfg_interval
        if not guid_t_min < guid_t_max:
            raise ValueError("cfg_interval should be (min, max) with min < max")

        t_half = t[: len(t) // 2]
        half_eps = torch.where(
            ((t_half >= guid_t_min) & (t_half <= guid_t_max)).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps),
            cond_eps,
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(
        self,
        x,
        t,
        y,
        cfg_scale,
        additional_model_forward: Callable,
        cfg_interval=(0, 1),
        cond: torch.Tensor | None = None,
    ):
        model_out = self.forward(x, t, y, cond=cond)
        try:
            ag_model_out = additional_model_forward(x, t, y, cond=cond)
        except TypeError:
            ag_model_out = additional_model_forward(x, t, y)

        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        if not guid_t_min < guid_t_max:
            raise ValueError("cfg_interval should be (min, max) with min < max")

        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps),
            eps,
        )
        return eps

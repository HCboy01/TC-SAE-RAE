from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2WithRegistersModel


class DinoClsSaeConditioner(nn.Module):
    """Extract DINO CLS token and convert it into SAE latent condition."""

    def __init__(
        self,
        encoder_config_path: str,
        dinov2_path: str,
        encoder_input_size: int,
        sae_ckpt_path: str,
        sae_src_path: str | None = None,
    ):
        super().__init__()
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.register_buffer("mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False)
        self.encoder_input_size = int(encoder_input_size)

        self.dino = Dinov2WithRegistersModel.from_pretrained(dinov2_path)
        self.dino.requires_grad_(False)

        sae_src = Path(sae_src_path) if sae_src_path is not None else Path(__file__).resolve().parent.parent
        if str(sae_src) not in sys.path:
            sys.path.insert(0, str(sae_src))
        from sae_local.model import Autoencoder  # type: ignore

        ckpt = torch.load(sae_ckpt_path, map_location="cpu", weights_only=False)
        sae_state = ckpt.get("sae_state_dict", ckpt)

        # TC-batch checkpoint: "activation" key is at top-level of ckpt,
        # and sae_state_dict has both flat activation.* keys AND
        # "activation"/"activation_state_dict" entries added by Autoencoder.state_dict().
        # from_state_dict strips activation.* then calls load_state_dict(strict=True),
        # which fails because activation buffers are missing. Use strict=False instead.
        try:
            self.sae = Autoencoder.from_state_dict(dict(sae_state), strict=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAE checkpoint from {sae_ckpt_path}: {e}"
            ) from e
        self.sae.requires_grad_(False)

    @property
    def cond_dim(self) -> int:
        return int(self.sae.latent_bias.numel())

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, _, h, w = images.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            images = nn.functional.interpolate(
                images,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        x = (images - self.mean.to(images.device)) / self.std.to(images.device)
        out = self.dino(x)
        cls = out.last_hidden_state[:, 0, :]
        latents, _ = self.sae.encode(cls)
        return latents

from typing import Callable

import torch
import torch.nn as nn


class BatchTopK(nn.Module):
    """Batch-level TopK sparse activation.

    During training, selects the top k * batch_size activations across the
    entire batch rather than k per sample. Each sample gets ~k active features
    on average, but the exact count varies per sample.

    This distributes gradient more evenly across all latents, significantly
    reducing dead features when training on low-diversity datasets (e.g. faces).

    During inference (eval mode), a stored EMA of the training threshold is
    used to gate activations, keeping sparsity consistent with training.

    Args:
        k: Target number of active features per sample (same as TopK.k).
        postact_fn: Applied to passing values (default: ReLU).
        ema_momentum: EMA update rate for the inference threshold (default: 0.01).
    """

    def __init__(
        self,
        k: int,
        postact_fn: Callable = nn.ReLU(),
        ema_momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn
        self.ema_momentum = ema_momentum
        # Stored during training; used at inference time instead of batch competition.
        self.register_buffer("threshold_ema", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size = x.shape[0]
            total_k = min(self.k * batch_size, x.numel())

            with torch.no_grad():
                flat = x.reshape(-1)
                threshold_val = torch.topk(flat, total_k, sorted=False).values.min().float()
                self.threshold_ema.lerp_(threshold_val, self.ema_momentum)
        else:
            threshold_val = self.threshold_ema

        # Gradient flows through postact_fn(x) for elements above threshold.
        return self.postact_fn(x) * (x >= threshold_val)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd[prefix + "k"] = self.k
        sd[prefix + "postact_fn"] = self.postact_fn.__class__.__name__
        sd[prefix + "ema_momentum"] = self.ema_momentum
        return sd

    @classmethod
    def from_state_dict(cls, state_dict: dict, strict: bool = True) -> "BatchTopK":
        k = int(state_dict["k"])
        ema_momentum = float(state_dict.get("ema_momentum", 0.01))
        postact_name = str(state_dict.get("postact_fn", "ReLU"))
        postact_cls = {"ReLU": nn.ReLU, "Identity": nn.Identity}.get(postact_name, nn.ReLU)
        obj = cls(k=k, postact_fn=postact_cls(), ema_momentum=ema_momentum)
        if "threshold_ema" in state_dict:
            obj.threshold_ema.copy_(
                state_dict["threshold_ema"]
                if isinstance(state_dict["threshold_ema"], torch.Tensor)
                else torch.tensor(float(state_dict["threshold_ema"]))
            )
        return obj

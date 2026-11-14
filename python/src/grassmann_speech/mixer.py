from collections.abc import Sequence

import torch
from torch import nn

from grassmann_speech.plucker import pair_indices, plucker, plucker_norm_sq, symmetric_norm_sq, symmetric_products, unit

# feature, its squared norm from dot products, and feature(b, a) / feature(a, b)
FEATURES = {"plucker": (plucker, plucker_norm_sq, -1.0), "symmetric": (symmetric_products, symmetric_norm_sq, 1.0)}


class GrassmannMixer(nn.Module):
    """Pairs reduced states at fixed offsets, encodes each pair as unit Plucker coordinates, averages, projects to d."""

    def __init__(
        self,
        d: int,
        r: int,
        offsets: Sequence[int],
        two_sided: bool,
        mean_first: bool = True,
        features: str = "plucker",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.offsets, self.two_sided, self.mean_first, self.eps = tuple(offsets), two_sided, mean_first, eps
        self.feature, self.norm_sq, self.flip = FEATURES[features]
        self.red = nn.Linear(d, r)
        self.plu = nn.Linear(r * (r - 1) // 2, d)
        i, j = pair_indices(r)
        self.register_buffer("i", i, persistent=False)
        self.register_buffer("j", j, persistent=False)

    def forward(self, h: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """valid (batch, length) marks real frames: a pair counts only when both ends are real, padding gets g = 0."""
        # Lagrange norms cancel in half precision and |z|^4 overflows fp16, so the mix runs in fp32 under autocast
        with torch.autocast(h.device.type, enabled=False):
            z = self.red(h.to(self.red.weight.dtype))
            weight = None if valid is None else valid[..., None].to(z.dtype)
            return self.anchored(z, weight) if self.mean_first else self.as_written(z, weight)

    def pairs(self, length: int):
        for delta in self.offsets:
            if delta < length:
                yield delta

    def as_written(self, z: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """2512.19428 as printed: project every unit pair, then average over the valid pairs of each position."""
        acc = z.new_zeros(*z.shape[:2], self.plu.out_features)
        count = z.new_zeros(*z.shape[:2], 1)
        for delta in self.pairs(z.shape[1]):
            both = 1.0 if weight is None else weight[:, :-delta] * weight[:, delta:]
            # Paper pairs (z_t, z_{t+delta}) at t; assigning the pair to t+delta instead makes it causal (docs/architecture.md, A1)
            q = both * self.plu(unit(self.feature(z[:, :-delta], z[:, delta:], self.i, self.j), self.eps))
            acc[:, delta:] += q
            count[:, delta:] += both
            if self.two_sided:
                acc[:, :-delta] += q
                count[:, :-delta] += both
        # Positions with no valid pair get g = 0 (docs/architecture.md, A3)
        return acc / count.clamp_min(1) * (count > 0)

    def anchored(self, z: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """Same output as as_written: all pairs at t share z_t and the feature is linear in the partner, so one feature per token."""
        partner = torch.zeros_like(z)
        count = z.new_zeros(*z.shape[:2], 1)
        for delta in self.pairs(z.shape[1]):
            earlier, later = z[:, :-delta], z[:, delta:]
            both = 1.0 if weight is None else weight[:, :-delta] * weight[:, delta:]
            # Norms from dot products lose precision only for near-parallel pairs, which the eps clamp caps anyway
            w = both * self.norm_sq(earlier, later).clamp_min(0).sqrt().clamp_min(self.eps).reciprocal()
            partner[:, delta:] += w * earlier
            count[:, delta:] += both
            if self.two_sided:
                partner[:, :-delta] += self.flip * w * later
                count[:, :-delta] += both
        return self.plu(self.feature(partner / count.clamp_min(1), z, self.i, self.j)) * (count > 0)

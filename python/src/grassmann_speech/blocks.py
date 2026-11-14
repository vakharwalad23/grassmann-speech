from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from grassmann_speech.mixer import GrassmannMixer


def feed_forward(d: int, d_ff: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))


class GrassmannBlock(nn.Module):
    """Post-LN block of 2512.19428 Sec 3.2: gated blend of h and the Grassmann mix, LayerNorm, FFN, LayerNorm."""

    def __init__(self, d: int, r: int, offsets: Sequence[int], d_ff: int, dropout: float, two_sided: bool, **mixer_kw):
        super().__init__()
        self.mixer = GrassmannMixer(d, r, offsets, two_sided, **mixer_kw)
        self.gate = nn.Linear(2 * d, d)
        self.ln_mix, self.ln_ffn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = feed_forward(d, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g = self.mixer(h)
        alpha = torch.sigmoid(self.gate(torch.cat([h, g], dim=-1)))
        h_hat = self.drop(self.ln_mix(alpha * h + (1 - alpha) * g))
        return self.ln_ffn(h_hat + self.drop(self.ffn(h_hat)))


class TransformerBlock(nn.Module):
    """Post-LN causal self-attention block, the paper's size-matched baseline (2512.19428 Sec 4.1)."""

    def __init__(self, d: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ln_att, self.ln_ffn = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = feed_forward(d, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, length, d = h.shape
        q, k, v = self.qkv(h).view(b, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        h = self.ln_att(h + self.drop(self.out(a.transpose(1, 2).reshape(b, length, d))))
        return self.ln_ffn(h + self.drop(self.ffn(h)))

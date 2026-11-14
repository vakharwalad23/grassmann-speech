from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from grassmann_speech.mixer import GrassmannMixer


def feed_forward(d: int, d_ff: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d), nn.Dropout(dropout))


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1, self.pw2 = nn.Conv1d(d, 2 * d, 1), nn.Conv1d(d, d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.glu(self.pw1(self.ln(x).transpose(1, 2)), dim=1)
        return self.drop(self.pw2(F.silu(self.bn(self.dw(y)))).transpose(1, 2))


class SelfAttention(nn.Module):
    """Full-context multi-head attention; plain, not ESPnet's relative-position variant (d^2 + 2d fewer params per block)."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, length, d = u.shape
        q, k, v = self.qkv(u).view(b, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.out(a.transpose(1, 2).reshape(b, length, d))


class GatedGrassmann(nn.Module):
    """Two-sided Grassmann mix with the paper's sigmoid gate as a residual branch, (1 - alpha)(g - u) (this repo's variant, docs/architecture.md)."""

    def __init__(self, d: int, r: int, offsets: Sequence[int]):
        super().__init__()
        self.mixer = GrassmannMixer(d, r, offsets, two_sided=True)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        g = self.mixer(u)
        alpha = torch.sigmoid(self.gate(torch.cat([u, g], dim=-1)))
        return (1 - alpha) * (g - u)


class ConformerBlock(nn.Module):
    """Macaron block of Gulati et al. 2020 (arXiv 2005.08100) with the sequence mixer passed in."""

    def __init__(self, d: int, d_ff: int, kernel: int, mixer: nn.Module, dropout: float):
        super().__init__()
        self.ff1, self.ff2 = feed_forward(d, d_ff, dropout), feed_forward(d, d_ff, dropout)
        self.ln_mix, self.mixer = nn.LayerNorm(d), mixer
        self.conv = ConvModule(d, kernel, dropout)
        self.ln_out = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.drop(self.mixer(self.ln_mix(x)))
        x = x + self.conv(x)
        return self.ln_out(x + 0.5 * self.ff2(x))


class Subsampling(nn.Module):
    """Two 3x3 stride-2 convolutions: 10 ms fbank frames to 40 ms encoder frames."""

    def __init__(self, n_mels: int, channels: int, d: int):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(1, channels, 3, 2), nn.Conv2d(channels, channels, 3, 2)
        self.out = nn.Linear(channels * (((n_mels - 1) // 2 - 1) // 2), d)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv2(F.relu(self.conv1(feats.unsqueeze(1)))))
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f))


class ConformerCTC(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module], n_mels: int, channels: int, d: int, vocab: int):
        super().__init__()
        self.frontend = Subsampling(n_mels, channels, d)
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(d, vocab)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.frontend(feats)
        for block in self.blocks:
            x = block(x)
        return self.head(x).log_softmax(dim=-1)


def conformer_ctc(
    layers: int, mixer: str, d: int = 256, d_ff: int = 1024, heads: int = 4, kernel: int = 31, r: int = 32,
    offsets: Sequence[int] = (1, 2, 4, 8, 12, 16), n_mels: int = 80, channels: int = 256, vocab: int = 5000, dropout: float = 0.1,
) -> ConformerCTC:
    """mixer is "attention" (arm A) or "grassmann" (arm B); defaults follow the e02 shape (docs/architecture.md)."""
    make = {"attention": lambda: SelfAttention(d, heads, dropout), "grassmann": lambda: GatedGrassmann(d, r, offsets)}[mixer]
    blocks = [ConformerBlock(d, d_ff, kernel, make(), dropout) for _ in range(layers)]
    return ConformerCTC(blocks, n_mels, channels, d, vocab)

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from grassmann_speech.mixer import GrassmannMixer

MIXERS = ("attention", "grassmann", "symmetric", "symmetric_signed", "none")


def feed_forward(d: int, d_ff: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d), nn.Dropout(dropout))


def rel_positions(length: int, d: int, like: torch.Tensor) -> torch.Tensor:
    """Sinusoids of relative positions length-1 down to -(length-1), shape (1, 2 length - 1, d), as ESPnet RelPositionalEncoding."""
    pos = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=like.device)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=like.device) * -(math.log(10000.0) / d))
    pe = torch.stack([torch.sin(pos * div), torch.cos(pos * div)], dim=-1).flatten(1)
    return pe[None].to(like.dtype)


def rel_shift(scores: torch.Tensor) -> torch.Tensor:
    """(..., T, 2T-1) scores against relative positions T-1..-(T-1) to (..., T, T) with out[i, j] = scores[i, T-1-i+j]."""
    *lead, t, n = scores.shape
    padded = torch.cat([scores.new_zeros(*lead, t, 1), scores], dim=-1).view(*lead, n + 1, t)
    return padded[..., 1:, :].reshape(*lead, t, n)[..., : n // 2 + 1]


class ConvModule(nn.Module):
    def __init__(self, d: int, kernel: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1, self.pw2 = nn.Conv1d(d, 2 * d, 1), nn.Conv1d(d, d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        y = F.glu(self.pw1(self.ln(x).transpose(1, 2)), dim=1)
        # Padded frames hold the pointwise biases by now; zeroed so edge frames see what they see unpadded (ESPnet mask_pad)
        y = y.masked_fill(~valid[:, None, :], 0.0)
        return self.drop(self.pw2(F.silu(self.bn(self.dw(y)))).transpose(1, 2))


class RelPositionAttention(nn.Module):
    """Multi-head attention with Transformer-XL relative positions, ESPnet rel_selfattn (latest): 5d^2 + 6d parameters."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q, self.k, self.v, self.out = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.pos = nn.Linear(d, d, bias=False)
        self.bias_u, self.bias_v = nn.Parameter(torch.empty(heads, d // heads)), nn.Parameter(torch.empty(heads, d // heads))
        nn.init.xavier_uniform_(self.bias_u)
        nn.init.xavier_uniform_(self.bias_v)

    def split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.heads, -1)

    def forward(self, u: torch.Tensor, valid: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, length, d = u.shape
        q, k, v, p = self.split(self.q(u)), self.split(self.k(u)), self.split(self.v(u)), self.split(self.pos(pos))
        position = rel_shift((q + self.bias_v).transpose(1, 2) @ p.permute(0, 2, 3, 1)) / math.sqrt(d // self.heads)
        mask = position.masked_fill(~valid[:, None, None, :], torch.finfo(position.dtype).min)
        a = F.scaled_dot_product_attention((q + self.bias_u).transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(a.transpose(1, 2).reshape(b, length, d))


class GatedGrassmann(nn.Module):
    """Two-sided Grassmann mix with the paper's sigmoid gate as a residual branch, (1 - alpha)(g - u) (this repo's variant, docs/architecture.md)."""

    def __init__(self, d: int, r: int, offsets: Sequence[int], features: str):
        super().__init__()
        self.mixer = GrassmannMixer(d, r, offsets, two_sided=True, features=features)
        self.gate = nn.Linear(2 * d, d)

    def forward(self, u: torch.Tensor, valid: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        g = self.mixer(u, valid)
        alpha = torch.sigmoid(self.gate(torch.cat([u, g.to(u.dtype)], dim=-1)))
        return (1 - alpha) * (g - u)


class NoMixer(nn.Module):
    def forward(self, u: torch.Tensor, valid: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u)


class ConformerBlock(nn.Module):
    """Macaron block of Gulati et al. 2020 (arXiv 2005.08100) with the sequence mixer passed in, laid out as ESPnet's EncoderLayer."""

    def __init__(self, d: int, d_ff: int, kernel: int, mixer: nn.Module, dropout: float):
        super().__init__()
        self.ff1, self.ff2 = feed_forward(d, d_ff, dropout), feed_forward(d, d_ff, dropout)
        self.ln_mix, self.mixer = nn.LayerNorm(d), mixer
        self.conv = ConvModule(d, kernel, dropout)
        self.ln_out = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.drop(self.mixer(self.ln_mix(x), valid, pos))
        x = x + self.conv(x, valid)
        return self.ln_out(x + 0.5 * self.ff2(x))


class Subsampling(nn.Module):
    """Two 3x3 stride-2 convolutions: 10 ms fbank frames to 40 ms encoder frames (ESPnet Conv2dSubsampling)."""

    def __init__(self, n_mels: int, channels: int, d: int):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(1, channels, 3, 2), nn.Conv2d(channels, channels, 3, 2)
        self.out = nn.Linear(channels * (((n_mels - 1) // 2 - 1) // 2), d)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv2(F.relu(self.conv1(feats.unsqueeze(1)))))
        b, c, t, f = x.shape
        return self.out(x.transpose(1, 2).reshape(b, t, c * f)), ((lengths - 1) // 2 - 1) // 2


class ConformerCTC(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module], n_mels: int, channels: int, d: int, vocab: int, dropout: float):
        super().__init__()
        self.frontend = Subsampling(n_mels, channels, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """feats (batch, frames, n_mels) with lengths in frames; returns CTC log-probs (batch, frames / 4, vocab) and their lengths."""
        x, lengths = self.frontend(feats, lengths)
        valid = torch.arange(x.shape[1], device=x.device) < lengths[:, None]
        # Input scaling by sqrt(d) and dropout on both streams follow ESPnet RelPositionalEncoding, for every arm
        pos = self.drop(rel_positions(x.shape[1], x.shape[2], x))
        x = self.drop(x * math.sqrt(x.shape[2]))
        for block in self.blocks:
            x = block(x, valid, pos)
        return self.head(self.ln(x)).log_softmax(dim=-1), lengths


def mixer_module(mixer: str, d: int, heads: int, r: int, offsets: Sequence[int], dropout: float) -> nn.Module:
    if mixer == "attention":
        return RelPositionAttention(d, heads, dropout)
    if mixer in ("grassmann", "symmetric", "symmetric_signed"):
        return GatedGrassmann(d, r, offsets, "plucker" if mixer == "grassmann" else mixer)
    if mixer == "none":
        return NoMixer()
    raise ValueError(f"mixer must be one of {MIXERS}, got {mixer}")


def conformer_ctc(
    layers: int, mixer: str, d: int = 256, d_ff: int = 1024, heads: int = 4, kernel: int = 31, r: int = 32,
    offsets: Sequence[int] = (1, 2, 4, 8, 12, 16), n_mels: int = 80, channels: int = 256, vocab: int = 5000, dropout: float = 0.1,
) -> ConformerCTC:
    """Defaults are the e02 shape (docs/architecture.md); mixer is one of MIXERS."""
    blocks = [ConformerBlock(d, d_ff, kernel, mixer_module(mixer, d, heads, r, offsets, dropout), dropout) for _ in range(layers)]
    return ConformerCTC(blocks, n_mels, channels, d, vocab, dropout)

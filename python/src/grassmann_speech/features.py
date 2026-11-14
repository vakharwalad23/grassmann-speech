import math

import numpy as np
import soxr
import torch
import torch.nn.functional as F
from torch import nn


def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    """Slaney mel scale (librosa default, htk=False): linear below 1 kHz, logarithmic above."""
    log_step = math.log(6.4) / 27.0
    return torch.where(hz < 1000.0, hz * 3.0 / 200.0, 15.0 + torch.log(hz.clamp_min(1e-10) / 1000.0) / log_step)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    log_step = math.log(6.4) / 27.0
    return torch.where(mel < 15.0, mel * 200.0 / 3.0, 1000.0 * torch.exp(log_step * (mel - 15.0)))


def mel_filters(n_fft: int, n_mels: int, sample_rate: int) -> torch.Tensor:
    """(n_fft // 2 + 1, n_mels) triangles with Slaney area norm, as librosa.filters.mel, which ESPnet's LogMel frontend uses."""
    fft_hz = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64)
    mel_hz = mel_to_hz(torch.linspace(0, hz_to_mel(torch.tensor(sample_rate / 2.0, dtype=torch.float64)).item(), n_mels + 2,
                                      dtype=torch.float64))
    width = mel_hz.diff()
    ramps = mel_hz[:, None] - fft_hz[None, :]
    lower, upper = -ramps[:-2] / width[:-1, None], ramps[2:] / width[1:, None]
    weights = torch.clamp(torch.minimum(lower, upper), min=0) * (2.0 / (mel_hz[2:] - mel_hz[:-2]))[:, None]
    return weights.T.float()


class LogMel(nn.Module):
    """ESPnet DefaultFrontend: STFT (Hann window, centered, reflect pad), power, mel, log clamped at 1e-10."""

    def __init__(self, n_fft: int, win: int, hop: int, n_mels: int, sample_rate: int):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.register_buffer("mel", mel_filters(n_fft, n_mels, sample_rate), persistent=False)

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wave (batch, samples) with lengths in samples; returns (batch, frames, n_mels) and lengths in frames."""
        with torch.autocast(wave.device.type, enabled=False):
            spec = torch.stft(wave.float(), self.n_fft, self.hop, self.window.shape[0], self.window, center=True, pad_mode="reflect",
                              return_complex=True)
            feats = (spec.abs().square().transpose(1, 2) @ self.mel).clamp_min(1e-10).log()
        return feats, lengths // self.hop + 1


def speed_perturb(wave: np.ndarray, factor: float, sample_rate: int) -> np.ndarray:
    """Tempo and pitch by factor, as sox speed in ESPnet's recipe: read as if sampled at factor x rate, resample to rate."""
    return wave if factor == 1.0 else soxr.resample(wave, sample_rate * factor, sample_rate)


def time_warp(x: torch.Tensor, window: int) -> torch.Tensor:
    """ESPnet time_warp on one utterance (frames, bins): stretch [0, c) to [0, w) and squeeze the rest, bicubic."""
    t = x.shape[0]
    if t - window <= window:
        return x
    center = int(torch.randint(window, t - window, (1,)))
    warped = int(torch.randint(center - window, center + window, (1,))) + 1
    img = x[None, None]
    left = F.interpolate(img[:, :, :center], (warped, x.shape[1]), mode="bicubic", align_corners=False)
    right = F.interpolate(img[:, :, center:], (t - warped, x.shape[1]), mode="bicubic", align_corners=False)
    return torch.cat([left, right], dim=2)[0, 0]


def mask_along(x: torch.Tensor, dim: int, max_width: int, count: int) -> torch.Tensor:
    """ESPnet mask_along_axis: count masks per utterance, width uniform in [0, max_width), set to zero."""
    b, size = x.shape[0], x.shape[dim]
    widths = torch.randint(0, max(1, max_width), (b, count, 1), device=x.device)
    starts = torch.randint(0, max(1, size - int(widths.max())), (b, count, 1), device=x.device)
    span = torch.arange(size, device=x.device)
    hit = ((starts <= span) & (span < starts + widths)).any(dim=1)
    return x.masked_fill(hit[:, :, None] if dim == 1 else hit[:, None, :], 0.0)


class SpecAugment(nn.Module):
    """ESPnet SpecAug as the LS-100 recipe sets it: time warp, frequency masks, time masks scaled to the padded length."""

    def __init__(self, warp_window: int, freq_masks: int, freq_width: int, time_masks: int, time_ratio: float):
        super().__init__()
        self.warp_window, self.freq_masks, self.freq_width, self.time_masks, self.time_ratio = (
            warp_window, freq_masks, freq_width, time_masks, time_ratio)

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return feats
        warped = feats.clone()
        for i, n in enumerate(lengths.tolist()):
            warped[i, :n] = time_warp(feats[i, :n], self.warp_window)
        out = mask_along(warped, 2, self.freq_width, self.freq_masks)
        return mask_along(out, 1, math.floor(feats.shape[1] * self.time_ratio), self.time_masks)

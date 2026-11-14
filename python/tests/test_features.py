import math

import numpy as np
import torch

from grassmann_speech.features import LogMel, SpecAugment, hz_to_mel, mel_filters, mel_to_hz, speed_perturb


def test_slaney_mel_scale():
    hz = torch.tensor([0.0, 500.0, 1000.0, 2000.0, 8000.0], dtype=torch.float64)
    assert torch.allclose(hz_to_mel(hz)[:3], torch.tensor([0.0, 7.5, 15.0], dtype=torch.float64))
    assert math.isclose(hz_to_mel(hz)[3].item(), 15 + math.log(2) * 27 / math.log(6.4))
    assert torch.allclose(mel_to_hz(hz_to_mel(hz)), hz)


def test_filters_have_unit_area_where_bins_resolve_them():
    w = mel_filters(512, 80, 16000)
    assert w.shape == (257, 80) and (w >= 0).all()
    area = w.sum(0) * 16000 / 512
    assert torch.allclose(area[40:], torch.ones(40), atol=0.05)


def test_logmel_frames_and_tone_position():
    t = torch.arange(16000) / 16000
    feats, frames = LogMel(512, 400, 160, 80, 16000)(torch.sin(2 * math.pi * 1000 * t)[None], torch.tensor([16000]))
    assert feats.shape == (1, 101, 80) and frames.tolist() == [101]
    assert feats[0, 50].argmax().item() in (25, 26)


def test_speed_perturb_scales_length():
    wave = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    assert abs(len(speed_perturb(wave, 1.1, 16000)) - 16000 / 1.1) <= 2
    assert abs(len(speed_perturb(wave, 0.9, 16000)) - 16000 / 0.9) <= 2
    assert speed_perturb(wave, 1.0, 16000) is wave


def test_specaugment_masks_only_in_training():
    torch.manual_seed(0)
    aug, feats = SpecAugment(5, 2, 27, 5, 0.05), torch.randn(4, 300, 80) + 10
    assert torch.equal(aug.eval()(feats, torch.tensor([300, 250, 200, 150])), feats)
    out = aug.train()(feats, torch.tensor([300, 250, 200, 150]))
    assert out.shape == feats.shape and (out == 0).any()

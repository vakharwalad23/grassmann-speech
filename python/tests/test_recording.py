import numpy as np

from grassmann_speech.recording import SR, distortions, lowpass, reverb


def tone(hz: float) -> np.ndarray:
    return np.sin(2 * np.pi * hz * np.arange(SR) / SR)


def test_lowpass_keeps_the_band_and_removes_above():
    assert np.allclose(lowpass(tone(1000), 4000), tone(1000), atol=1e-9)
    assert np.abs(lowpass(tone(6000), 4000)).max() < 1e-9


def test_reverb_of_an_impulse_is_direct_path_plus_equal_energy_tail():
    impulse = np.zeros(SR)
    impulse[0] = 1.0
    out = reverb(impulse, 0.5, np.random.default_rng(0))
    assert len(out) == SR and out[0] == 1.0
    assert np.isclose((out[1:] ** 2).sum(), 1.0)


def test_distortions_keep_length_and_report_padding():
    wave = 0.1 * tone(300)
    for name, (out, pad) in distortions(wave).items():
        assert len(out) == len(wave) + 2 * pad, name
    assert distortions(wave)["digital_zeros_1s_each_side"][1] == SR

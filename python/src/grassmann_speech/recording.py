"""Synthetic stand-ins for a live recording: macOS speech, and the gain, mic response, noise, reverb, band limit, clipping and digital
silence a capture chain adds."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def say(text: str, out: Path, voice: str | None = None) -> np.ndarray:
    """macOS text to speech at 16 kHz mono; speech-like audio with no download."""
    voice_args = ["-v", voice] if voice else []
    subprocess.run(["say", *voice_args, "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True)
    wave, sr = sf.read(out, dtype="float64")
    assert sr == SR
    return wave


def add_floor(wave: np.ndarray, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """White noise at a fixed level; a microphone never delivers exact zeros, synthetic speech does."""
    return wave + 10 ** (dbfs / 20) * rng.standard_normal(len(wave))


def add_noise(wave: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    return wave + np.sqrt((wave**2).mean()) * 10 ** (-snr_db / 20) * rng.standard_normal(len(wave))


def reverb(wave: np.ndarray, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Direct path plus an exponential noise tail that decays 60 dB in rt60 s, tail energy equal to the direct path (DRR 0 dB)."""
    t = np.arange(int(rt60 * SR)) / SR
    tail = rng.standard_normal(len(t)) * 10 ** (-3 * t / rt60)
    tail[0] = 0.0
    rir = tail / np.sqrt((tail**2).sum())
    rir[0] = 1.0
    return np.convolve(wave, rir)[: len(wave)]


def lowpass(wave: np.ndarray, hz: float) -> np.ndarray:
    spec = np.fft.rfft(wave)
    spec[int(hz / (SR / 2) * (len(spec) - 1)):] = 0
    return np.fft.irfft(spec, n=len(wave))


def distortions(wave: np.ndarray, seed: int = 2) -> dict[str, tuple[np.ndarray, int]]:
    """Each distorted clip with the number of padding samples before the speech."""
    rng = np.random.default_rng(seed)
    peak, pad = np.abs(wave).max(), np.zeros(SR)
    return {
        "gain_-20db": (wave * 0.1, 0),
        "mic_fir_1_-0.6_0.2": (np.convolve(wave, [1.0, -0.6, 0.2])[: len(wave)], 0),
        "white_noise_snr_20db": (add_noise(wave, 20, rng), 0),
        "white_noise_snr_10db": (add_noise(wave, 10, rng), 0),
        "reverb_rt60_0.5s": (reverb(wave, 0.5, rng), 0),
        "lowpass_4khz": (lowpass(wave, 4000), 0),
        "clip_at_30pct_peak": (np.clip(wave, -0.3 * peak, 0.3 * peak), 0),
        "digital_zeros_1s_each_side": (np.concatenate([pad, wave, pad]), SR),
    }


VOICES = ("Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa", "Aman", "Eddy (English (US))")
SENTENCES = (
    "please turn off the lights in the kitchen before you go to bed",
    "the weather today is cold and windy with a chance of rain in the evening",
    "we should meet at the library after lunch to finish the report",
    "my brother bought a new bicycle and rode it all the way to the river",
    "can you tell me how long it takes to walk to the station from here",
    "she opened the window and listened to the birds singing in the garden",
    "the doctor said that a short walk every morning would help him sleep better",
    "i forgot my umbrella on the train and had to buy another one",
    "they painted the old house white and planted roses along the fence",
    "remember to call your mother on sunday and ask her about the trip",
    "the children laughed when the dog jumped into the swimming pool",
    "this machine listens to my voice and writes down every word i say",
)


def synthetic_set(tmp: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """8 voices x 12 sentences, -60 dBFS floor; ids are voice-sentence so the scorer groups by voice."""
    rng, clips, refs = np.random.default_rng(1), {}, {}
    for voice in VOICES:
        for k, sentence in enumerate(SENTENCES):
            uid = f"{voice.split()[0].lower()}-{k:02d}"
            clips[uid] = add_floor(say(sentence, tmp / f"{uid}.wav", voice), -60, rng)
            refs[uid] = sentence.upper()
    return clips, refs

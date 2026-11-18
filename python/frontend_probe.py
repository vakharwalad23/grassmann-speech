"""What a recording does to the e02 log-mel features: level, noise floor, bandwidth, distance from the LibriSpeech training statistics,
and which normalization cancels which distortion (gain, mic response, noise, reverb, band limit, clipping, digital silence)."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

from grassmann_speech.features import LogMel
from grassmann_speech.recording import SR, add_floor, distortions, say

HOP = 160
LOG_FLOOR = float(np.log(1e-10))
TEXT = ("The quick brown fox jumps over the lazy dog. She sells sea shells by the sea shore, "
        "and the shells she sells are surely sea shells.")


def load(path: Path) -> np.ndarray:
    wave, sr = sf.read(path, dtype="float64", always_2d=True)
    wave = wave.mean(axis=1)
    return wave if sr == SR else soxr.resample(wave, sr, SR)


def training_stats(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean and std of the training features, as ESPnet's global_mvn stores them in the model state dict."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return state["normalize.mean"].double().numpy(), state["normalize.std"].double().numpy()


def logmel(wave: np.ndarray) -> np.ndarray:
    """(frames, 80) on the e02 frontend: n_fft 512, window 400, hop 160, Slaney mel, log clamped at 1e-10."""
    x = torch.from_numpy(wave).float()[None]
    feats, _ = LogMel(512, 400, HOP, 80, SR)(x, torch.tensor([x.shape[1]]))
    return feats[0].double().numpy()


def describe(wave: np.ndarray) -> dict:
    """Level, clipping, digital zeros, DC, noise floor and speech level of 25 ms frames, 99 percent energy bandwidth."""
    frames = np.lib.stride_tricks.sliding_window_view(wave, 400)[::HOP]
    energy_db = 10 * np.log10((frames**2).mean(axis=1) + 1e-20)
    floor_db, speech_db = np.percentile(energy_db, 10), np.percentile(energy_db, 95)
    loud = frames[energy_db > speech_db - 20] * np.hanning(400)
    power = (np.abs(np.fft.rfft(loud, axis=1)) ** 2).mean(axis=0)
    cumulative = np.cumsum(power) / power.sum()
    return {"seconds": len(wave) / SR, "peak_dbfs": float(20 * np.log10(np.abs(wave).max() + 1e-20)),
            "rms_dbfs": float(10 * np.log10((wave**2).mean() + 1e-20)), "clipped_fraction": float((np.abs(wave) >= 0.999).mean()),
            "exact_zero_fraction": float((wave == 0).mean()), "dc": float(wave.mean()), "noise_floor_db": float(floor_db),
            "speech_level_db": float(speech_db), "snr_estimate_db": float(speech_db - floor_db),
            "bandwidth_99_hz": float(np.searchsorted(cumulative, 0.99) * SR / 400)}


def feature_report(feats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    """Mean over bins of |utterance bin mean - training bin mean| / training bin std, and how often the log floor is hit."""
    offset = (feats.mean(axis=0) - mean) / std
    return {"train_offset_mean_abs_std": float(np.abs(offset).mean()), "train_offset_low_bins_0_19": float(offset[:20].mean()),
            "train_offset_high_bins_60_79": float(offset[60:].mean()), "log_floor_fraction": float((feats <= LOG_FLOOR + 1e-6).mean())}


def normalizations(mean: np.ndarray, std: np.ndarray) -> dict:
    return {
        "scalar_per_utterance": lambda f: (f - f.mean()) / f.std(),
        "global_mvn_librispeech": lambda f: (f - mean) / std,
        "cmvn_per_utterance": lambda f: (f - f.mean(axis=0)) / f.std(axis=0),
    }


def distortion_table(clean: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict:
    """RMS distance, over speech frames, between the normalized features of the distorted and the clean clip."""
    ref = logmel(clean)
    n, table = ref.shape[0], {}
    for name, (wave, pad) in distortions(clean).items():
        feats = logmel(wave)
        start = pad // HOP
        row = {"features": feature_report(feats, mean, std)}
        for norm_name, norm in normalizations(mean, std).items():
            a, b = norm(feats)[start + 3: start + n - 3], norm(ref)[3: n - 3]
            row[norm_name] = float(np.sqrt(((a - b) ** 2).mean()))
        table[name] = row
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, help="a recording to describe; without it, the distortion table on a synthetic clip")
    parser.add_argument("--stats", type=Path, default=Path("vendor/espnet/egs2/librispeech_100/asr1/exp/asr_attention_s0/valid.cer_ctc.ave.pth"),
                        help="ESPnet e02 checkpoint holding normalize.mean and normalize.std")
    parser.add_argument("--out", type=Path, default=Path("reports/frontend_probe.json"))
    args = parser.parse_args()
    mean, std = training_stats(args.stats)
    if args.wav:
        wave = load(args.wav)
        report = {"recording": str(args.wav), "audio": describe(wave), "features": feature_report(logmel(wave), mean, std)}
    else:
        with tempfile.TemporaryDirectory() as tmp:
            clean = add_floor(say(TEXT, Path(tmp) / "say.wav"), -60, np.random.default_rng(1))
        report = {"clip": {"text": TEXT, "floor_dbfs": -60}, "audio": describe(clean),
                  "features": feature_report(logmel(clean), mean, std), "distortions": distortion_table(clean, mean, std)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

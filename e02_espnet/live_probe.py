"""An e02 ESPnet checkpoint on synthetic speech under live-capture distortions, greedy CTC, with and without inference-time feature fixes."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from espnet2.tasks.asr import ASRTask
from grassmann_speech.recording import VOICES, distortions, synthetic_set
from grassmann_speech.wer import paired_bootstrap, per_utterance, totals

LOG_FLOOR = float(np.log(1e-10))
WORD_START = "\u2581"


def load_model(exp: Path, tmp: Path) -> tuple[torch.nn.Module, list[str], torch.Tensor]:
    """ESPnet model; its global_mvn stats file stayed on the training host, so the stats come from the checkpoint buffers."""
    checkpoint = exp / "valid.cer_ctc.ave.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mean, std = state["normalize.mean"].double().numpy(), state["normalize.std"].double().numpy()
    np.savez(tmp / "feats_stats.npz", count=np.array(1.0), sum=mean, sum_square=mean**2 + std**2)
    config = yaml.safe_load((exp / "config.yaml").read_text())
    config["normalize_conf"]["stats_file"] = str(tmp / "feats_stats.npz")
    (tmp / "config.yaml").write_text(yaml.safe_dump(config))
    model, _ = ASRTask.build_model_from_file(tmp / "config.yaml", checkpoint, "cpu")
    return model.eval(), config["token_list"], state["normalize.mean"].float()


def fixed(fix: str, feats: torch.Tensor, train_mean: torch.Tensor) -> torch.Tensor:
    """level_match shifts all bins by one number, mean_match moves each bin's mean to the training mean; frames at the log floor are skipped."""
    real = feats[0].mean(-1) > LOG_FLOOR + 5
    if fix == "level_match":
        return feats + (train_mean.mean() - feats[0, real].mean())
    if fix == "mean_match":
        return feats - feats[0, real].mean(0) + train_mean
    return feats


@torch.no_grad()
def transcribe(model: torch.nn.Module, tokens: list[str], wave: np.ndarray, fix: str, train_mean: torch.Tensor) -> str:
    speech = torch.from_numpy(wave).float()[None]
    feats, lengths = model._extract_feats(speech, torch.tensor([speech.shape[1]]))
    feats, lengths = model.normalize(fixed(fix, feats, train_mean), lengths)
    encoded = model.encoder(feats, lengths)[0]
    ids = torch.unique_consecutive(model.ctc.argmax(encoded)[0]).tolist()
    return "".join(tokens[i] for i in ids if i != 0).replace(WORD_START, " ").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path, default=Path("vendor/espnet/egs2/librispeech_100/asr1/exp/asr_attention_s0"))
    parser.add_argument("--out", type=Path, default=Path("reports/e02_live_probe.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        model, tokens, train_mean = load_model(args.exp, Path(tmp))
        clips, refs = synthetic_set(Path(tmp))
    variants = {uid: {"clean": (wave, 0), **distortions(wave, seed=k)} for k, (uid, wave) in enumerate(clips.items())}
    report, clean = {"exp": str(args.exp), "voices": list(VOICES), "utterances": len(clips), "conditions": {}}, None
    for condition in variants[next(iter(variants))]:
        rows = {}
        for fix in ("none", "level_match", "mean_match"):
            hyps = {uid: transcribe(model, tokens, variants[uid][condition][0], fix, train_mean) for uid in clips}
            rows[fix] = per_utterance(refs, hyps)
            if fix == "none":
                report.setdefault("hyps", {})[condition] = hyps
            if condition == "clean" and fix == "none":
                clean = rows[fix]
        report["conditions"][condition] = {
            fix: {**totals(r), **({"vs_none": paired_bootstrap(r, rows["none"])} if fix != "none" else {"vs_clean": paired_bootstrap(r, clean)})}
            for fix, r in rows.items()}
        print(condition, {fix: round(v["wer"], 1) for fix, v in report["conditions"][condition].items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

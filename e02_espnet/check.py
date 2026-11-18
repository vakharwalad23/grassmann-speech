"""Checks the ESPnet track on CPU before any GPU hour: arm sizes, frontend parity with the own trainer, one tiny train and decode."""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import yaml
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.frontend.default import DefaultFrontend
from grassmann_speech.conformer import conformer_ctc
from grassmann_speech.features import LogMel

from encoder import MixerConformerEncoder

HERE = Path(__file__).resolve().parent
# Own-trainer sizes, unit-tested in python/tests/test_conformer.py
EXPECTED = {"attention": 26_958_472, "grassmann": 26_021_992, "symmetric": 26_021_992, "symmetric_signed": 26_021_992,
            "no_mixer": 26_945_032}
CTC_HEAD = 256 * 5000 + 5000


def encoder(arm: str, **override) -> torch.nn.Module:
    config = yaml.safe_load((HERE / "conf" / f"{arm}.yaml").read_text())
    kind = ConformerEncoder if config["encoder"] == "conformer" else MixerConformerEncoder
    return kind(80, **{**config["encoder_conf"], **override})


def check_sizes() -> None:
    for arm, expected in EXPECTED.items():
        got = sum(p.numel() for p in encoder(arm).parameters()) + CTC_HEAD
        own = conformer_ctc(15, {"no_mixer": "none"}.get(arm, arm), d_ff=1344 if arm == "no_mixer" else 1024)
        assert got == expected == sum(p.numel() for p in own.parameters()), (arm, got)
        print(f"{arm}: {got:,} params, same as the own trainer")


def check_frontend() -> None:
    wave = torch.from_numpy(np.random.default_rng(0).standard_normal((2, 32000)).astype(np.float32)) * 0.1
    lengths = torch.tensor([32000, 24000])
    theirs, their_lens = DefaultFrontend(fs=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80)(wave, lengths)
    ours, our_lens = LogMel(512, 400, 160, 80, 16000)(wave, lengths)
    assert torch.equal(their_lens, our_lens)
    diff = (theirs[1, : our_lens[1]] - ours[1, : our_lens[1]]).abs().max().item()
    assert diff < 1e-3, diff
    print(f"log-mel: max abs difference to ESPnet's frontend {diff:.2e}")


def tiny_recipe(work: Path) -> list[str]:
    """Kaldi-style data dir, unigram BPE and token list as asr.sh writes them (blank, unk, pieces, sos/eos)."""
    rng, words = np.random.default_rng(0), ["THE", "CAT", "SAT", "ON", "A", "MAT", "DOG", "RAN"]
    lines = {"wav.scp": [], "text": [], "speech_shape": []}
    for u in range(8):
        uid, text = f"10{u % 2}-1-{u:04d}", " ".join(rng.choice(words, size=4))
        sf.write(work / f"{uid}.flac", 0.1 * rng.standard_normal(16000 + 800 * u), 16000)
        lines["wav.scp"].append(f"{uid} {work / f'{uid}.flac'}")
        lines["text"].append(f"{uid} {text}")
        lines["speech_shape"].append(f"{uid} {16000 + 800 * u}")
    for name, rows in lines.items():
        (work / name).write_text("\n".join(rows) + "\n")
    (work / "plain.txt").write_text("\n".join(r.split(" ", 1)[1] for r in lines["text"]) + "\n")
    spm.SentencePieceTrainer.train(input=str(work / "plain.txt"), model_prefix=str(work / "bpe"), vocab_size=20, model_type="unigram",
                                   character_coverage=1.0)
    pieces = [line.split("\t")[0] for line in (work / "bpe.vocab").read_text().splitlines()[3:]]
    tokens = ["<blank>", "<unk>", *pieces, "<sos/eos>"]
    (work / "tokens.txt").write_text("\n".join(tokens) + "\n")
    sp = spm.SentencePieceProcessor(model_file=str(work / "bpe.model"))
    (work / "text_shape.bpe").write_text("".join(f"{r.split(' ', 1)[0]} {len(sp.encode(r.split(' ', 1)[1]))},{len(tokens)}\n"
                                                 for r in lines["text"]))
    data = [f"{work}/wav.scp,speech,sound", f"{work}/text,text,text"]
    shapes = [f"{work}/speech_shape", f"{work}/text_shape.bpe"]
    return [*(a for d in data for a in ("--train_data_path_and_name_and_type", d)), *(a for s in shapes for a in ("--train_shape_file", s)),
            *(a for d in data for a in ("--valid_data_path_and_name_and_type", d)), *(a for s in shapes for a in ("--valid_shape_file", s)),
            "--token_type", "bpe", "--bpemodel", f"{work}/bpe.model", "--token_list", f"{work}/tokens.txt"]


def check_train_and_decode(arm: str, work: Path, data_args: list[str]) -> None:
    """Runs the real ESPnet entry points through launch.py, as asr.sh does with --python."""
    exp = work / f"exp_{arm}"
    subprocess.run([str(HERE / "python"), "-m", "espnet2.bin.asr_train", "--config", str(HERE / "conf" / f"{arm}.yaml"), *data_args,
                    "--encoder_conf", "num_blocks=2", "--normalize", "none", "--use_amp", "false", "--ngpu", "0", "--num_workers", "0",
                    "--max_epoch", "1", "--batch_bins", "400000", "--accum_grad", "1", "--output_dir", str(exp)], check=True)
    decode = work / "decode.yaml"
    decode.write_text("lm_weight: 0.0\nctc_weight: 1.0\nbeam_size: 1\n")
    subprocess.run([str(HERE / "python"), "-m", "espnet2.bin.asr_inference", "--config", str(decode), "--ngpu", "0",
                    "--data_path_and_name_and_type", f"{work}/wav.scp,speech,sound", "--asr_train_config", str(exp / "config.yaml"),
                    "--asr_model_file", str(exp / "1epoch.pth"), "--output_dir", str(work / f"decode_{arm}")], check=True)
    hyps = (work / f"decode_{arm}" / "1best_recog" / "text").read_text().splitlines()
    assert len(hyps) == 8, hyps
    print(f"{arm}: trained 1 epoch and decoded 8 utterances through launch.py")


def main() -> None:
    check_sizes()
    check_frontend()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        data_args = tiny_recipe(work)
        for arm in sys.argv[1:] or EXPECTED:
            check_train_and_decode(arm, work, data_args)


if __name__ == "__main__":
    main()

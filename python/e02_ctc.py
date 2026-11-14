import argparse
import faulthandler
import json
import shutil
import signal
import time
import tomllib
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from grassmann_speech.conformer import ConformerCTC, conformer_ctc
from grassmann_speech.env import device, environment
from grassmann_speech.features import LogMel, SpecAugment
from grassmann_speech.librispeech import (
    SAMPLE_RATE,
    SPLITS,
    Utterances,
    collate,
    numel_batches,
    read_jsonl,
    scan,
    sealed_manifest,
    synthetic_corpus,
    train_tokenizer,
    write_jsonl,
)
from grassmann_speech.wer import per_utterance, totals

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_suffix(".toml")


def load_config(quick: bool) -> dict:
    cfg = tomllib.loads(CONFIG.read_text())
    for section, override in cfg.pop("quick").items():
        if quick:
            cfg[section].update(override)
    return cfg


def paths(cfg: dict, quick: bool) -> dict[str, Path]:
    work = ROOT / "work" / "e02" / ("quick" if quick else "own")
    return {"work": work, "librispeech": work / "LibriSpeech" if quick else ROOT / cfg["data"]["librispeech"],
            "corpora": work / "corpora" if quick else ROOT / "corpora", "reports": work / "reports" if quick else ROOT / "reports" / "e02_own",
            "tokenizer": work / "tokenizer.json", "mvn": work / "mvn.pt"}


class Frontend(torch.nn.Module):
    """Wave to normalized log-mel, in ESPnet's order: log-mel, SpecAugment (training only), global mean-variance norm, zeroed padding."""

    def __init__(self, cfg: dict, mvn: dict[str, torch.Tensor]):
        super().__init__()
        f, s = cfg["features"], cfg["specaug"]
        self.logmel = LogMel(f["n_fft"], f["win"], f["hop"], f["n_mels"], SAMPLE_RATE)
        self.specaug = SpecAugment(s["warp_window"], s["freq_masks"], s["freq_width"], s["time_masks"], s["time_ratio"])
        self.register_buffer("mean", mvn["mean"])
        self.register_buffer("std", mvn["std"])

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats, lengths = self.logmel(wave, lengths)
        feats = (self.specaug(feats, lengths) - self.mean) / self.std
        valid = torch.arange(feats.shape[1], device=feats.device) < lengths[:, None]
        return feats * valid[..., None], lengths


def prepare(cfg: dict, p: dict[str, Path], quick: bool) -> None:
    """Split lists, sealed dev and test manifests, tokenizer and global MVN stats; reads the extracted LibriSpeech tree only."""
    if quick:
        synthetic_corpus(p["librispeech"], 12, 0)
    for split in SPLITS:
        rows = scan(p["librispeech"], split)
        write_jsonl(p["work"] / f"{split}.jsonl", rows)
        if split != cfg["data"]["train"]:
            out = p["corpora"] / f"librispeech-{split}" / "manifest.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(sealed_manifest(p["librispeech"], split, rows), indent=1))
        print(f"{split}: {len(rows)} utterances, {sum(r['samples'] for r in rows) / SAMPLE_RATE / 3600:.2f} h")
    train = read_jsonl(p["work"] / f"{cfg['data']['train']}.jsonl")
    tok = train_tokenizer([r["text"] for r in train], cfg["data"]["vocab"], p["tokenizer"])
    assert quick or tok.get_vocab_size() == cfg["data"]["vocab"], tok.get_vocab_size()
    torch.save(mvn_stats(cfg, p, train), p["mvn"])


def mvn_stats(cfg: dict, p: dict[str, Path], train: list[dict]) -> dict[str, torch.Tensor]:
    f = cfg["features"]
    logmel = LogMel(f["n_fft"], f["win"], f["hop"], f["n_mels"], SAMPLE_RATE)
    picked = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[: cfg["data"]["mvn_utterances"]]
    total, total_sq, frames = 0.0, 0.0, 0
    for i in picked.tolist():
        wave = torch.from_numpy(sf.read(p["librispeech"] / train[i]["path"], dtype="float32")[0])
        feats = logmel(wave[None], torch.tensor([len(wave)]))[0][0].double()
        total, total_sq, frames = total + feats.sum(0), total_sq + feats.square().sum(0), frames + feats.shape[0]
    mean = total / frames
    return {"mean": mean.float(), "std": (total_sq / frames - mean.square()).clamp_min(1e-20).sqrt().float()}


def items(cfg: dict, p: dict[str, Path], split: str, tok: Tokenizer, factors: list[float]) -> list[dict]:
    rows = read_jsonl(p["work"] / f"{split}.jsonl")
    ids = [e.ids for e in tok.encode_batch([r["text"] for r in rows])]
    out = [{**r, "factor": f, "samples": int(r["samples"] / f), "tokens": t} for r, t in zip(rows, ids, strict=True) for f in factors]
    # The recipe drops utterances over 30 s after speed perturbation (asr.sh --max_wav_duration)
    return [it for it in out if it["samples"] <= cfg["data"]["max_seconds"] * SAMPLE_RATE]


def loader(dataset_items: list[dict], batches: list[list[int]], p: dict[str, Path], workers: int) -> DataLoader:
    return DataLoader(Utterances(dataset_items, p["librispeech"]), batch_sampler=batches, collate_fn=collate, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


def build(cfg: dict, arm: str, vocab: int) -> ConformerCTC:
    m, a = cfg["model"], cfg["arms"][arm]
    return conformer_ctc(m["layers"], a["mixer"], d=m["d"], d_ff=a["d_ff"], heads=m["heads"], kernel=m["kernel"], r=m["r"],
                         offsets=m["offsets"], n_mels=cfg["features"]["n_mels"], vocab=vocab, dropout=m["dropout"])


def warmup_lr(step: int, warmup: int) -> float:
    """ESPnet WarmupLR as a factor on the peak lr: linear up to warmup steps, then inverse square root; step counts from 1."""
    return warmup**0.5 * min(step**-0.5, step * warmup**-1.5)


def ctc_loss(model: ConformerCTC, frontend: Frontend, batch: dict, dev: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summed CTC loss over the batch divided by its size, as ESPnet's CTC with length_normalized_loss false."""
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        feats, lengths = frontend(batch["wave"].to(dev), batch["wave_lens"].to(dev))
        log_probs, out_lens = model(feats, lengths)
    loss = F.ctc_loss(log_probs.float().transpose(0, 1), batch["tokens"].to(dev), out_lens, batch["token_lens"].to(dev),
                      reduction="sum", zero_infinity=True)
    return loss / len(batch["ids"]), log_probs, out_lens


def greedy(log_probs: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    """Best path, repeats merged, blanks (id 0) dropped."""
    out = []
    for seq, n in zip(log_probs.argmax(-1).tolist(), lengths.tolist(), strict=True):
        out.append([t for k, t in enumerate(seq[:n]) if t != 0 and (k == 0 or t != seq[k - 1])])
    return out


@torch.no_grad()
def evaluate(model: ConformerCTC, frontend: Frontend, data: DataLoader, tok: Tokenizer, refs: dict[str, str], dev: str) -> tuple[dict, dict]:
    model.eval()
    frontend.eval()
    loss, count, hyps = 0.0, 0, {}
    for batch in data:
        batch_loss, log_probs, out_lens = ctc_loss(model, frontend, batch, dev)
        loss, count = loss + batch_loss.item() * len(batch["ids"]), count + len(batch["ids"])
        hyps.update(zip(batch["ids"], tok.decode_batch(greedy(log_probs, out_lens)), strict=True))
    model.train()
    frontend.train()
    return {"loss": loss / count, **totals(per_utterance({u: refs[u] for u in hyps}, hyps))}, hyps


def train(cfg: dict, p: dict[str, Path], arm: str, seed: int, dev: str) -> None:
    t, tok = cfg["train"], Tokenizer.from_file(str(p["tokenizer"]))
    vocab, run = tok.get_vocab_size(), p["work"] / f"{arm}_s{seed}"
    torch.manual_seed(seed)
    model, frontend = build(cfg, arm, vocab).to(dev), Frontend(cfg, torch.load(p["mvn"])).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: warmup_lr(step + 1, t["warmup"]))
    train_items = items(cfg, p, cfg["data"]["train"], tok, cfg["data"]["speed_factors"])
    valid_items = [it for split in cfg["data"]["valid"] for it in items(cfg, p, split, tok, [1.0])]
    train_batches = numel_batches(train_items, t["batch_bins"], vocab)
    valid_data = loader(valid_items, numel_batches(valid_items, t["batch_bins"], vocab), p, t["workers"])
    refs = {it["id"]: it["text"] for it in valid_items}
    report_path = p["reports"] / f"{arm}_s{seed}.json"
    record = {"arm": arm, "seed": seed, "params": sum(q.numel() for q in model.parameters()), "vocab": vocab, "config": cfg,
              "device": dev, "env": environment(ROOT), "train_utterances": len(train_items), "batches_per_epoch": len(train_batches),
              "epochs": [], "status": "running"}
    if (run / "last.pt").exists():
        state = torch.load(run / "last.pt")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        record["epochs"] = state["epochs"]
        print(f"resumed after epoch {len(record['epochs'])}")
    run.mkdir(parents=True, exist_ok=True)
    p["reports"].mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for epoch in range(len(record["epochs"]) + 1, t["epochs"] + 1):
        order = torch.randperm(len(train_batches), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()
        epoch_start, total, audio, skipped = time.perf_counter(), 0.0, 0, 0
        for i, batch in enumerate(loader(train_items, [train_batches[k] for k in order], p, t["workers"])):
            loss = ctc_loss(model, frontend, batch, dev)[0]
            (loss / t["accum"]).backward()
            total, audio = total + loss.item(), audio + int(batch["wave_lens"].sum())
            if (i + 1) % t["accum"] == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                # ESPnet skips the update on a non-finite gradient norm
                if torch.isfinite(norm):
                    opt.step()
                    sched.step()
                else:
                    skipped += 1
                opt.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)
        train_s = time.perf_counter() - epoch_start
        valid = evaluate(model, frontend, valid_data, tok, refs, dev)[0]
        row = {"epoch": epoch, "train_loss": total / len(train_batches), "valid_loss": valid["loss"], "valid_wer": valid["wer"],
               "lr": sched.get_last_lr()[0], "skipped_steps": skipped, "train_s": train_s, "audio_s_per_s": audio / SAMPLE_RATE / train_s,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else None}
        record["epochs"].append(row)
        torch.save(model.state_dict(), run / f"epoch{epoch}.pt")
        keep = {r["epoch"] for r in sorted(record["epochs"], key=lambda r: r["valid_wer"])[: t["keep_best"]]}
        for old in run.glob("epoch*.pt"):
            if int(old.stem[5:]) not in keep:
                old.unlink()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epochs": record["epochs"]},
                   run / "last.pt")
        record.update(wall_s=time.perf_counter() - start, best_epochs=sorted(keep))
        report_path.write_text(json.dumps(record, indent=1))
        print(f"{arm} s{seed} epoch {epoch}: train {row['train_loss']:.2f} valid {row['valid_loss']:.2f} WER {row['valid_wer']:.2f} "
              f"{row['audio_s_per_s']:.0f} audio s/s", flush=True)
    record["status"] = "trained"
    report_path.write_text(json.dumps(record, indent=1))


def averaged(run: Path, epochs: list[int]) -> dict[str, torch.Tensor]:
    """Mean of the kept checkpoints (the recipe averages the 10 best by validation); integer buffers keep the first value."""
    states = [torch.load(run / f"epoch{e}.pt") for e in epochs]
    return {k: sum(s[k] for s in states) / len(states) if v.is_floating_point() else v for k, v in states[0].items()}


def decode(cfg: dict, p: dict[str, Path], arm: str, seed: int, dev: str) -> None:
    tok, report_path = Tokenizer.from_file(str(p["tokenizer"])), p["reports"] / f"{arm}_s{seed}.json"
    record = json.loads(report_path.read_text())
    model, frontend = build(cfg, arm, tok.get_vocab_size()).to(dev), Frontend(cfg, torch.load(p["mvn"])).to(dev)
    model.load_state_dict(averaged(p["work"] / f"{arm}_s{seed}", record["best_epochs"]))
    out = p["reports"] / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    record["test"] = {}
    for split in cfg["data"]["test"]:
        test_items = items(cfg, p, split, tok, [1.0])
        refs = {it["id"]: it["text"] for it in test_items}
        data = loader(test_items, numel_batches(test_items, cfg["train"]["batch_bins"], tok.get_vocab_size()), p, cfg["train"]["workers"])
        start = time.perf_counter()
        scores, hyps = evaluate(model, frontend, data, tok, refs, dev)
        (out / f"{split}.ref").write_text("".join(f"{u} {refs[u]}\n" for u in sorted(refs)))
        (out / f"{split}.hyp").write_text("".join(f"{u} {hyps[u]}\n" for u in sorted(hyps)))
        record["test"][split] = {**scores, "decode_s": time.perf_counter() - start}
        print(f"{arm} s{seed} {split}: WER {scores['wer']:.2f} ({scores['errors']} / {scores['words']}), CER {scores['cer']:.2f}")
    record["status"] = "done"
    report_path.write_text(json.dumps(record, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="e02 own trainer: Conformer-CTC on LibriSpeech-100, attention vs Grassmann vs controls")
    parser.add_argument("step", choices=("prepare", "train", "decode", "quick"))
    parser.add_argument("--arm", default="attention")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    faulthandler.register(signal.SIGUSR1)
    quick = args.step == "quick"
    cfg, dev = load_config(quick), device()
    p = paths(cfg, quick)
    if quick:
        shutil.rmtree(p["work"], ignore_errors=True)
        prepare(cfg, p, quick)
        for arm in cfg["arms"]:
            train(cfg, p, arm, args.seed, dev)
            decode(cfg, p, arm, args.seed, dev)
    elif args.step == "prepare":
        prepare(cfg, p, quick)
    elif args.step == "train":
        train(cfg, p, args.arm, args.seed, dev)
    else:
        decode(cfg, p, args.arm, args.seed, dev)


if __name__ == "__main__":
    main()

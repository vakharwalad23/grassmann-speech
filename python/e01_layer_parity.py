import argparse
import faulthandler
import json
import math
import signal
import time
import tomllib
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from grassmann_speech.env import device, environment
from grassmann_speech.lm import LanguageModel, grassmann_lm, transformer_lm
from grassmann_speech.mixer import GrassmannMixer
from grassmann_speech.plucker import dots

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_suffix(".toml")
REPORTS = ROOT / "reports" / "e01_layer_parity"
CACHE = ROOT / "work" / "e01"
OFFSETS = (1, 2, 4, 8, 12, 16)
ARMS = {
    "transformer": lambda dropout: transformer_lm(6, dropout=dropout),
    "grassmann": lambda dropout: grassmann_lm([OFFSETS] * 6, dropout=dropout),
    "grassmann_one_offset": lambda dropout: grassmann_lm([(o,) for o in OFFSETS], dropout=dropout),
    "control_symmetric": lambda dropout: grassmann_lm([OFFSETS] * 6, dropout=dropout, features="symmetric"),
}


def token_ids(data: dict, split: str) -> torch.Tensor:
    cached = CACHE / f"{split}.pt"
    if cached.exists():
        return torch.load(cached)
    parquet = hf_hub_download(data["dataset"], f"{data['subset']}/{split}-00000-of-00001.parquet", repo_type="dataset",
                              revision=data["dataset_revision"])
    tokenizer = Tokenizer.from_file(hf_hub_download(data["tokenizer"], "tokenizer.json", revision=data["tokenizer_revision"]))
    lines = [line for line in pq.read_table(parquet, columns=["text"]).column("text").to_pylist() if line.strip()]
    # No [CLS]/[SEP]: the stream is one running text cut into fixed blocks (2512.19428 Sec 4.1)
    ids = torch.tensor([i for enc in tokenizer.encode_batch(lines, add_special_tokens=False) for i in enc.ids], dtype=torch.long)
    cached.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, cached)
    return ids


def blocks(ids: torch.Tensor, block: int) -> torch.Tensor:
    """Contiguous, non-overlapping blocks of block+1 tokens; inputs are [:-1], targets [1:]."""
    n = (ids.numel() - 1) // block
    return ids[: n * block + 1].unfold(0, block + 1, block)


def loss_on(model: LanguageModel, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())


@torch.no_grad()
def perplexity(model: LanguageModel, data: torch.Tensor, batch: int, device: str, limit: int | None) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, data.shape[0] if limit is None else min(limit, data.shape[0]), batch):
        chunk = data[start : start + batch].to(device)
        total += loss_on(model, chunk).item() * chunk[:, 1:].numel()
        count += chunk[:, 1:].numel()
    model.train()
    return math.exp(total / count)


@torch.no_grad()
def pair_geometry(model: LanguageModel, batch: torch.Tensor) -> dict:
    """Per layer and offset: quantiles of sin(angle) between z_{t-delta} and z_t; near 0 means the plane is ill-defined."""
    inputs, hooks = {}, []
    for index, block in enumerate(model.blocks):
        if isinstance(getattr(block, "mixer", None), GrassmannMixer):
            hooks.append(block.mixer.register_forward_pre_hook(lambda _m, args, index=index: inputs.__setitem__(index, args[0])))
    model.eval()
    model(batch[:, :-1])
    model.train()
    for hook in hooks:
        hook.remove()
    stats = {}
    for index, h in inputs.items():
        mixer = model.blocks[index].mixer
        z = mixer.red(h)
        for delta in mixer.pairs(z.shape[1]):
            aa, bb, ab = dots(z[:, :-delta], z[:, delta:])
            sin = ((aa * bb - ab * ab).clamp_min(0) / (aa * bb).clamp_min(1e-12)).sqrt().flatten().float().cpu()
            q = torch.quantile(sin, torch.tensor([0.01, 0.5, 0.99]))
            stats[f"layer{index}_offset{delta}"] = {"sin_p01": q[0].item(), "sin_p50": q[1].item(), "sin_p99": q[2].item(),
                                                    "frac_below_0.05": (sin < 0.05).float().mean().item()}
    return stats


def train_run(arm: str, seed: int, cfg: dict, splits: dict, device: str, quick: bool, out: Path) -> dict:
    t = cfg["train"]
    torch.manual_seed(seed)
    model = ARMS[arm](t["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    epochs, limit = (1, 5 * t["batch"]) if quick else (t["epochs"], None)
    order = torch.Generator().manual_seed(seed)
    record = {"arm": arm, "seed": seed, "params": sum(p.numel() for p in model.parameters()), "config": cfg, "device": device,
              "env": environment(ROOT), "tokens": {k: int(v.numel()) for k, v in splits["ids"].items()}, "epochs": [], "status": "running"}
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train = splits["train"][torch.randperm(splits["train"].shape[0], generator=order)]
        train = train if limit is None else train[:limit]
        epoch_start, total = time.perf_counter(), 0.0
        for i in range(0, train.shape[0], t["batch"]):
            loss = loss_on(model, train[i : i + t["batch"]].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        train_s = time.perf_counter() - epoch_start
        row = {"epoch": epoch, "train_loss": total / math.ceil(train.shape[0] / t["batch"]), "train_s": train_s,
               "tokens_per_s": train[:, 1:].numel() / train_s,
               "val_ppl": perplexity(model, splits["validation"], t["batch"], device, limit),
               "test_ppl": perplexity(model, splits["test"], t["batch"], device, limit)}
        record["epochs"].append(row)
        best = min(record["epochs"], key=lambda r: r["val_ppl"])
        # Paper reports best validation PPL over training; test PPL is read at that epoch, never used for selection
        record.update(best_epoch=best["epoch"], best_val_ppl=best["val_ppl"], test_ppl_at_best=best["test_ppl"],
                      wall_s=time.perf_counter() - start)
        out.write_text(json.dumps(record, indent=1))
        print(f"{arm} s{seed} epoch {epoch}: loss {row['train_loss']:.3f} val {row['val_ppl']:.1f} test {row['test_ppl']:.1f} "
              f"{row['tokens_per_s']:.0f} tok/s", flush=True)
    if arm != "transformer":
        record["pair_geometry"] = pair_geometry(model, splits["validation"][: t["batch"]].to(device))
    record["status"] = "done"
    out.write_text(json.dumps(record, indent=1))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="e01: Grassmann vs Transformer LM on Wikitext-2 at 13M params (layer parity)")
    parser.add_argument("--runs", help="comma list of arm:seed to run, default every [[run]] in the config")
    parser.add_argument("--quick", action="store_true", help="1 epoch of 5 batches per run into work/, checks the code paths only")
    args = parser.parse_args()
    faulthandler.register(signal.SIGUSR1)
    cfg = tomllib.loads(CONFIG.read_text())
    dev = device()
    ids = {split: token_ids(cfg["data"], split) for split in ("train", "validation", "test")}
    splits = {"ids": ids, **{split: blocks(v, cfg["train"]["block"]) for split, v in ids.items()}}
    print({split: int(v.numel()) for split, v in ids.items()}, "tokens")
    wanted = None if args.runs is None else {tuple(r.split(":")) for r in args.runs.split(",")}
    out_dir = CACHE / "quick" if args.quick else REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in cfg["run"]:
        arm, seed = run["arm"], run["seed"]
        out = out_dir / f"{arm}_s{seed}.json"
        if wanted is not None and (arm, str(seed)) not in wanted:
            continue
        if not args.quick and out.exists() and json.loads(out.read_text())["status"] == "done":
            print(f"{arm} s{seed}: done, skipped")
            continue
        r = train_run(arm, seed, cfg, splits, dev, args.quick, out)
        print(f"{arm} s{seed}: {r['params']} params, best val {r['best_val_ppl']:.1f} at epoch {r['best_epoch']}, "
              f"test {r['test_ppl_at_best']:.1f}, {r['wall_s'] / 3600:.2f} h")


if __name__ == "__main__":
    main()

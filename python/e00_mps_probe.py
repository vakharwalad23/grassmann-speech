import argparse
import faulthandler
import gc
import json
import signal
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from grassmann_speech.blocks import GrassmannBlock, TransformerBlock
from grassmann_speech.conformer import conformer_ctc
from grassmann_speech.env import environment
from grassmann_speech.lm import grassmann_lm, transformer_lm

ROOT = Path(__file__).resolve().parents[1]
# 14-core M3 Pro FP32: 2 x 1792 ALUs x ~1.3-1.4 GHz = 4.6-5.0 TFLOPS, planning value, not a measured peak
PEAK_TFLOPS = 4.8
OFFSETS = (1, 2, 4, 8, 12, 16)
UTTERANCE_S = 12.7
# e01: 30 epochs x ~2.4M Wikitext-2 tokens (token count assumed); e02: 70 epochs x 3 speed copies x 100.6 h (ESPnet LS-100 recipe)
E01_TOKENS = 30 * 2.4e6
E02_AUDIO_S = 70 * 3 * 100.6 * 3600
FULL = {"lengths": (128, 256, 512, 1024, 2048, 4096), "tokens": 16384, "warmup": 3, "runs": 20, "lm_batch": 32, "lm_len": 128,
        "asr_batches": (8, 16), "asr_layers": 15, "asr_s": UTTERANCE_S}
QUICK = {"lengths": (64, 128), "tokens": 256, "warmup": 1, "runs": 2, "lm_batch": 2, "lm_len": 128,
         "asr_batches": (2,), "asr_layers": 2, "asr_s": 2.0}
BLOCK_KEYS = ("arm", "length", "batch", "step_s", "tokens_per_s", "pct_peak", "mem_mb", "error")
LM_KEYS = ("arm", "step_s", "tokens_per_s", "pct_peak", "mem_mb", "e01_run_h", "error")
ASR_KEYS = ("arm", "batch", "step_s", "audio_s_per_s", "pct_peak", "mem_mb", "e02_run_h", "error")

Step = Callable[[], None]


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def train_step(model: torch.nn.Module, loss_fn: Callable[[], torch.Tensor], autocast: bool, device: str) -> Step:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step() -> None:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
            loss = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return step


def block_step(kind: str, length: int, batch: int, device: str) -> Step:
    torch.manual_seed(0)
    block = {
        "attention": lambda: TransformerBlock(256, 4, 1024, 0.1),
        "grassmann_as_written": lambda: GrassmannBlock(256, 32, OFFSETS, 1024, 0.1, two_sided=False, mean_first=False),
        "grassmann_mean_first": lambda: GrassmannBlock(256, 32, OFFSETS, 1024, 0.1, two_sided=False),
        "grassmann_two_sided": lambda: GrassmannBlock(256, 32, OFFSETS, 1024, 0.1, two_sided=True),
    }[kind]().to(device)
    x = torch.randn(batch, length, 256, device=device)
    return train_step(block, lambda: block(x).square().mean(), False, device)


def lm_step(kind: str, batch: int, length: int, device: str) -> Step:
    torch.manual_seed(0)
    model = {
        "transformer": lambda: transformer_lm(6),
        "grassmann_as_written": lambda: grassmann_lm([OFFSETS] * 6, mean_first=False),
        "grassmann_mean_first": lambda: grassmann_lm([OFFSETS] * 6),
    }[kind]().to(device)
    tokens = torch.randint(0, 30522, (batch, length + 1), device=device)
    return train_step(model, lambda: F.cross_entropy(model(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten()), False, device)


def asr_step(kind: str, batch: int, layers: int, seconds: float, device: str) -> Step:
    torch.manual_seed(0)
    mixer, precision = kind.rsplit("_", 1)
    model = conformer_ctc(layers, mixer).to(device)
    feats = torch.randn(batch, int(seconds * 100), 80, device=device)
    # ~2.7 words/s in LibriSpeech test-clean (52,576 words / 5.4 h), ~1.5 BPE-5000 tokens per word (assumed)
    n_targets = max(1, int(seconds * 4))
    targets = torch.randint(1, 5000, (batch, n_targets), device=device)
    feat_lengths = torch.full((batch,), feats.shape[1], dtype=torch.long, device=device)
    target_lengths = torch.full((batch,), n_targets, dtype=torch.long, device=device)

    def loss() -> torch.Tensor:
        log_probs, out_lengths = model(feats, feat_lengths)
        return F.ctc_loss(log_probs.transpose(0, 1).float(), targets, out_lengths, target_lengths)

    return train_step(model, loss, precision == "bf16", device)


def matmul_flops(make: Callable[[str], Step]) -> int:
    step = make("cpu")
    with FlopCounterMode(display=False) as counter:
        step()
    return counter.get_total_flops()


def memory_mb(step: Step, device: str) -> float | None:
    if device != "mps":
        return None
    gc.collect()
    torch.mps.empty_cache()
    before = torch.mps.driver_allocated_memory()
    step()
    sync(device)
    # Freed blocks stay cached in the driver, so the delta after one step approximates the step's peak working set
    return (torch.mps.driver_allocated_memory() - before) / 2**20


def measure(arms: dict[str, Callable[[str], Step]], device: str, cfg: dict) -> dict[str, dict]:
    out, steps = {}, {}
    for name, make in arms.items():
        try:
            steps[name] = make(device)
            out[name] = {"mem_mb": memory_mb(steps[name], device)}
            for _ in range(cfg["warmup"]):
                steps[name]()
            sync(device)
        except RuntimeError as err:
            out[name] = {"error": str(err).splitlines()[0]}
            steps.pop(name, None)
    times = {name: [] for name in steps}
    # Interleaved A/B: each round times every arm once, so drift hits all arms alike
    for _ in range(cfg["runs"]):
        for name, step in steps.items():
            start = time.perf_counter()
            step()
            sync(device)
            times[name].append(time.perf_counter() - start)
    for name, values in times.items():
        out[name].update(step_s=statistics.median(values), step_s_min=min(values), step_s_max=max(values))
    return out


def with_flops(results: dict[str, dict], flops: dict[str, int]) -> dict[str, dict]:
    for name, row in results.items():
        row["matmul_gflop"] = flops[name] / 1e9
        if "step_s" in row:
            row["tflops"] = flops[name] / row["step_s"] / 1e12
            row["pct_peak"] = 100 * row["tflops"] / PEAK_TFLOPS
    return results


def block_sweep(device: str, cfg: dict) -> list[dict]:
    kinds = ("attention", "grassmann_as_written", "grassmann_mean_first", "grassmann_two_sided")
    rows = []
    for length in cfg["lengths"]:
        batch = max(1, cfg["tokens"] // length)
        arms = {k: (lambda dev, k=k, length=length, batch=batch: block_step(k, length, batch, dev)) for k in kinds}
        # FLOPs scale with batch, so count at batch 1 on CPU and multiply
        flops = {k: batch * matmul_flops(lambda dev, k=k, length=length: block_step(k, length, 1, dev)) for k in kinds}
        for name, row in with_flops(measure(arms, device, cfg), flops).items():
            rows.append({"arm": name, "length": length, "batch": batch, **row})
            if "step_s" in row:
                rows[-1]["tokens_per_s"] = batch * length / row["step_s"]
    return rows


def lm_steps(device: str, cfg: dict) -> list[dict]:
    batch, length = cfg["lm_batch"], cfg["lm_len"]
    kinds = ("transformer", "grassmann_as_written", "grassmann_mean_first")
    arms = {k: (lambda dev, k=k: lm_step(k, batch, length, dev)) for k in kinds}
    flops = {k: batch * matmul_flops(lambda dev, k=k: lm_step(k, 1, length, dev)) for k in kinds}
    rows = []
    for name, row in with_flops(measure(arms, device, cfg), flops).items():
        rows.append({"arm": name, "batch": batch, "length": length, **row})
        if "step_s" in row:
            rows[-1]["tokens_per_s"] = batch * length / row["step_s"]
            rows[-1]["e01_run_h"] = E01_TOKENS / rows[-1]["tokens_per_s"] / 3600
    return rows


def asr_steps(device: str, cfg: dict) -> list[dict]:
    layers, seconds = cfg["asr_layers"], cfg["asr_s"]
    kinds = ("attention_fp32", "grassmann_fp32", "attention_bf16", "grassmann_bf16")
    flops_one = {k: matmul_flops(lambda dev, k=k: asr_step(k.replace("bf16", "fp32"), 1, layers, seconds, dev)) for k in kinds}
    rows = []
    for batch in cfg["asr_batches"]:
        arms = {k: (lambda dev, k=k, batch=batch: asr_step(k, batch, layers, seconds, dev)) for k in kinds}
        for name, row in with_flops(measure(arms, device, cfg), {k: batch * f for k, f in flops_one.items()}).items():
            rows.append({"arm": name, "batch": batch, "layers": layers, "utterance_s": seconds, **row})
            if "step_s" in row:
                rows[-1]["audio_s_per_s"] = batch * seconds / row["step_s"]
                rows[-1]["e02_run_h"] = E02_AUDIO_S / rows[-1]["audio_s_per_s"] / 3600
    return rows


def ctc_parity(device: str) -> dict:
    torch.manual_seed(0)
    logits = torch.randn(316, 4, 5000)
    targets = torch.randint(1, 5000, (4, 60))
    lengths, target_lengths = torch.full((4,), 316), torch.full((4,), 60)

    def loss_and_grad(dev: str) -> tuple[float, torch.Tensor]:
        x = logits.detach().to(dev, copy=True).requires_grad_()
        loss = F.ctc_loss(x.log_softmax(-1), targets.to(dev), lengths.to(dev), target_lengths.to(dev))
        loss.backward()
        return loss.item(), x.grad.cpu()

    try:
        (ref, ref_grad), (got, grad) = loss_and_grad("cpu"), loss_and_grad(device)
    except RuntimeError as err:
        return {"error": str(err).splitlines()[0]}
    return {"loss_cpu": ref, "loss_device": got, "loss_abs_diff": abs(ref - got), "grad_max_abs_diff": (ref_grad - grad).abs().max().item()}


def print_rows(title: str, rows: list[dict], keys: tuple[str, ...]) -> None:
    print(f"\n{title}\n" + " | ".join(keys))
    for row in rows:
        print(" | ".join(f"{row[k]:.4g}" if isinstance(row.get(k), float) else str(row.get(k, "-")) for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description="e00: MPS speed, memory and ctc_loss probe for e01 and e02 planning")
    parser.add_argument("--quick", action="store_true", help="tiny sizes, seconds, no report file; checks the code paths only")
    args = parser.parse_args()
    faulthandler.register(signal.SIGUSR1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    env = environment(ROOT)
    if not args.quick and "AC Power" not in env.get("power", ""):
        raise SystemExit(f"e00 times a Mac GPU on AC power, got: {env.get('power', env['os'])}")
    cfg = QUICK if args.quick else FULL
    path = None if args.quick else ROOT / "reports" / "e00_mps_probe" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    print(json.dumps(env, indent=1))
    start = time.perf_counter()
    report = {"env": env, "device": device, "config": cfg, "peak_tflops_assumed": PEAK_TFLOPS}
    sections = (
        ("ctc_parity", lambda: ctc_parity(device), None),
        ("blocks", lambda: block_sweep(device, cfg), BLOCK_KEYS),
        ("lm", lambda: lm_steps(device, cfg), LM_KEYS),
        ("asr", lambda: asr_steps(device, cfg), ASR_KEYS),
    )
    for name, run, keys in sections:
        report[name] = run()
        report["wall_s"] = time.perf_counter() - start
        if keys is None:
            print(f"\n{name}: {report[name]}")
        else:
            print_rows(name, report[name], keys)
        # Saved after every section so a killed run keeps what finished
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1))
    print(f"\nwall {report['wall_s']:.0f} s" + (f", report {path.relative_to(ROOT)}" if path else ""))


if __name__ == "__main__":
    main()

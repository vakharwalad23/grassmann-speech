"""Summarize an ESPnet train.log: wall time, peak GPU memory, per-epoch losses and valid CER, 10 best by the 3-decimal logged CER."""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

RESULT = re.compile(
    r"\] (\S+ \S+),\d+ \(trainer:\d+\) INFO: (\d+)epoch results: \[train\].*?loss=([\d.]+),.*?"
    r"time=([^,]+), total_count.*?\[valid\] loss_ctc=([\d.]+), cer_ctc=([\d.]+)"
)
MEM = re.compile(r"gpu_max_cached_mem_GB=([\d.]+)")
STARTED = re.compile(r"^# Started at (.+)$", re.MULTILINE)


def epochs(log: str) -> list[dict]:
    return [
        {"epoch": int(e), "end_utc": t, "train_loss": float(tr), "train_time": tt, "valid_loss": float(vl), "valid_cer_ctc": float(c)}
        for t, e, tr, tt, vl, c in RESULT.findall(log)
    ]


def summary(log: str, keep: int = 10) -> dict:
    rows = epochs(log)
    start = datetime.strptime(STARTED.search(log).group(1).replace(" UTC", ""), "%a %b %d %H:%M:%S %Y")
    end = datetime.strptime(rows[-1]["end_utc"], "%Y-%m-%d %H:%M:%S")
    best = sorted(rows, key=lambda r: r["valid_cer_ctc"])[:keep]
    return {
        "epochs_done": len(rows),
        "wall_train_h": round((end - start).total_seconds() / 3600, 3),
        "gpu_max_cached_mem_GB": max(float(m) for m in MEM.findall(log)),
        "best_epochs_by_logged_cer": sorted(r["epoch"] for r in best),
        "best_valid_cer_ctc": best[0]["valid_cer_ctc"],
        "per_epoch": rows,
    }


def main() -> None:
    if len(sys.argv) == 1:
        demo = (
            "# Started at Sun Nov 15 12:39:51 UTC 2026\n"
            "[h] 2026-11-15 12:48:15,688 (trainer:392) INFO: 1epoch results: [train] iter_time=1, loss_ctc=120.8, loss=120.8, "
            "train_time=0.7, time=7 minutes and 20.19 seconds, total_count=2330, gpu_max_cached_mem_GB=19.9, "
            "[valid] loss_ctc=157.3, cer_ctc=1.000, loss_att=nan\n"
        )
        s = summary(demo)
        assert s["epochs_done"] == 1 and s["best_epochs_by_logged_cer"] == [1] and s["gpu_max_cached_mem_GB"] == 19.9
        assert abs(s["wall_train_h"] - 0.140) < 1e-3, s["wall_train_h"]
        print("summary.py self-check ok")
    else:
        print(json.dumps(summary(Path(sys.argv[1]).read_text()), indent=1))


if __name__ == "__main__":
    main()

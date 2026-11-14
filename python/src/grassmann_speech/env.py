import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch


def shell(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout.strip()


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def machine() -> dict:
    if platform.system() == "Darwin":
        return {
            "chip": shell("sysctl -n machdep.cpu.brand_string"),
            "gpu_cores": shell("system_profiler SPDisplaysDataType | awk -F': ' '/Total Number of Cores/ {print $2}'"),
            "ram_bytes": int(shell("sysctl -n hw.memsize")),
            "os": "macOS " + shell("sw_vers -productVersion") + " (" + shell("sw_vers -buildVersion") + ")",
            "power": shell("pmset -g batt | head -1"),
        }
    return {
        "chip": shell("awk -F': ' '/model name/ {print $2; exit}' /proc/cpuinfo"),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "ram_bytes": int(shell("awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo")),
        "os": platform.platform(),
    }


def environment(root: Path) -> dict:
    """What every measured number records : date, commit, machine, OS, power, torch."""
    return {
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        # Untracked files (a sweep's own reports) do not change what ran; only tracked edits mark the stamp dirty (e01)
        "commit": shell(f"git -C {root} rev-parse --short HEAD") + ("-dirty" if shell(f"git -C {root} status --porcelain --untracked-files=no") else ""),
        **machine(),
        "torch": torch.__version__,
    }

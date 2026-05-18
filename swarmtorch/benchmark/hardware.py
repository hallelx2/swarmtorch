"""Hardware fingerprint for reproducibility.

Every benchmark result carries a snapshot of the machine that produced
it — CPU model, GPU model, RAM, OS, Python version, PyTorch + CUDA
versions. This lets us compare runs across Kaggle / Colab / Jules /
local machines later without guessing what produced what.

The fingerprint is:

* embedded into every ``RunResult.meta["hardware"]`` so the JSON files
  are self-describing;
* printed as a banner at the top of every benchmark script so the
  Kaggle / Colab log opens with a clear identification of the runtime.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from functools import lru_cache
from typing import Any

import torch


def _read_cpuinfo() -> str:
    """Best-effort CPU model string across OSes."""
    # Linux (Kaggle, Colab, Jules):
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # macOS:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except (OSError, subprocess.SubprocessError):
        pass
    # Windows (and fallback):
    proc = platform.processor()
    return proc or platform.machine() or "unknown"


def _read_total_ram_mb() -> int:
    """Total RAM in MB."""
    # Linux:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(re.findall(r"\d+", line)[0])
                    return kb // 1024
    except OSError:
        pass
    # Windows / macOS fallbacks: try psutil if available, else 0.
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        return 0


def _git_sha() -> str | None:
    """Best-effort git commit SHA so result JSONs are tied to a code state."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.decode().strip()
    except (OSError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=1)
def hardware_info() -> dict[str, Any]:
    """Collect a hardware + software fingerprint.

    Cached because the values don't change within a process. Callers
    can rely on this being free to call repeatedly.
    """
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        # Machine identity
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        # Reproducibility -- ties this run to a code state.
        "git_sha": _git_sha(),
        # CPU
        "cpu_model": _read_cpuinfo(),
        "cpu_count_logical": os.cpu_count() or 0,
        "ram_mb": _read_total_ram_mb(),
        # Python / PyTorch
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda or "none",
        # GPU
        "cuda_available": cuda_available,
    }
    if cuda_available:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_mem_mb"] = int(props.total_memory / (1024 * 1024))
        info["gpu_capability"] = f"{props.major}.{props.minor}"
    else:
        info["gpu_name"] = None
        info["gpu_count"] = 0
        info["gpu_total_mem_mb"] = 0
        info["gpu_capability"] = None
    return info


def print_banner(extra: dict[str, Any] | None = None) -> None:
    """Print a hardware identification banner to stdout.

    Loud and easy to spot in long Kaggle / Colab logs. Idempotent —
    safe to call from every script entry point.
    """
    info = hardware_info()
    print("=" * 72)
    print("  swarmtorch benchmark run")
    print("=" * 72)
    print(f"  hostname        : {info['hostname']}")
    print(f"  git sha         : {info['git_sha'] or '(not a git checkout)'}")
    print(f"  os              : {info['os']}")
    print(f"  platform        : {info['platform']}")
    print(f"  cpu             : {info['cpu_model']}")
    print(
        f"  cpu cores       : {info['cpu_count_logical']} logical"
    )
    print(f"  ram             : {info['ram_mb']} MB" if info["ram_mb"] else "  ram             : unknown")
    print(f"  python          : {info['python']} ({info['python_implementation']})")
    print(f"  torch           : {info['torch']}")
    print(f"  torch built w/  : CUDA {info['torch_compiled_cuda']}")
    if info["cuda_available"]:
        print(f"  gpu             : {info['gpu_name']} (x{info['gpu_count']})")
        print(f"  gpu memory      : {info['gpu_total_mem_mb']} MB")
        print(f"  gpu capability  : sm_{info['gpu_capability']}")
    else:
        print("  gpu             : (none -- CPU-only run)")
    if extra:
        print("  -" * 25)
        for k, v in extra.items():
            print(f"  {k:<16}: {v}")
    print("=" * 72, flush=True)

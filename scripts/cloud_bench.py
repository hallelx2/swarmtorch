"""One-command orchestrator for cloud benchmark runs (Jules / Colab / Kaggle).

Runs the CPU-friendly Stage 4 sweeps in sequence with sane defaults and
no external dependencies beyond what ``pip install -e ".[benchmark,cmaes]"``
already installs. Use this instead of the Makefile when running on a
machine where ``make`` may not be on PATH (Jules slim images, Colab
notebooks, Windows without scoop, etc.).

Usage:

    # Default: ablations + HPO + synthetic at 5 seeds, max_fe=5000
    python scripts/cloud_bench.py

    # GPU machine? Add the headline GPU benchmark too:
    python scripts/cloud_bench.py --include-gpu

    # Quick smoke (~5 minutes) to verify the pipeline works:
    python scripts/cloud_bench.py --quick

    # Pick specific sweeps:
    python scripts/cloud_bench.py --sweeps ablation-init bench-hpo

After the run, commit results/ to a branch and open a PR if you're
running this through Jules. The Jules prompt for that workflow is in
AGENTS.md.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each sweep is a (name, [argv]) pair.  argv is what we'd pass to subprocess
# *without* the python executable -- we prepend sys.executable at run time so
# the sweep uses whatever venv the user is in.
SWEEPS: dict[str, list[str]] = {
    "ablation-init": [
        "scripts/run_ablations.py",
        "--ablation",
        "init_strategy",
        "--output-dir",
        "results/ablations/init",
        "--algorithms",
        "PSO",
        "CA",
        "TLBO",
        "CMAES",
    ],
    "ablation-swarm": [
        "scripts/run_ablations.py",
        "--ablation",
        "swarm_size",
        "--output-dir",
        "results/ablations/swarm",
        "--algorithms",
        "PSO",
        "CMAES",
    ],
    "bench-hpo": [
        "scripts/run_hpo.py",
        "--output-dir",
        "results/hpo",
        "--n-trials",
        "20",
    ],
    "bench-synthetic": [
        "scripts/run_synthetic.py",
        "--output-dir",
        "results/synthetic",
        "--max-fe",
        "5000",
        "--dims",
        "10",
        "50",
        "200",
    ],
    # Optional, only when --include-gpu is set.
    "bench-gpu": [
        "scripts/run_gpu_vs_numpy.py",
        "--output-dir",
        "results/gpu_vs_numpy",
        "--max-fe",
        "20000",
    ],
    # Real-NN training tasks (MNIST + CIFAR). Heavy on CPU, fast on GPU.
    "bench-training": [
        "scripts/run_training.py",
        "--output-dir",
        "results/training",
        "--max-fe",
        "3000",
    ],
    "bench-training-quick": [
        "scripts/run_training.py",
        "--output-dir",
        "results/training_quick",
        "--quick",
        "--max-fe",
        "500",
    ],
    # Smoke variants for --quick mode.
    "bench-synthetic-quick": [
        "scripts/run_synthetic.py",
        "--output-dir",
        "results/synthetic_quick",
        "--seeds",
        "0",
        "1",
        "--max-fe",
        "300",
        "--dims",
        "10",
        "--functions",
        "sphere",
        "rastrigin",
        "--algorithms",
        "PSO",
        "CMAES",
        "Adam",
    ],
    "bench-gpu-quick": [
        "scripts/run_gpu_vs_numpy.py",
        "--quick",
        "--output-dir",
        "results/gpu_quick",
    ],
}


def _run(name: str, argv: list[str], dry_run: bool) -> bool:
    cmd = [sys.executable, *argv]
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n========== [{name}] ==========")
    print(f"$ {pretty}")
    if dry_run:
        return True
    t0 = time.perf_counter()
    try:
        subprocess.check_call(cmd, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"!! [{name}] FAILED with exit code {e.returncode}")
        return False
    print(f"-- [{name}] done in {time.perf_counter() - t0:.1f}s")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweeps",
        nargs="+",
        choices=sorted(SWEEPS.keys()),
        default=None,
        help="Pick specific sweeps. Default: ablation-init, ablation-swarm, "
        "bench-hpo, bench-synthetic.",
    )
    p.add_argument(
        "--include-gpu",
        action="store_true",
        help="Also run the GPU-vs-NumPy benchmark. Skips automatically if "
        "torch.cuda.is_available() is False.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run the full standardised suite: ablations + HPO + synthetic + "
        "training + GPU benchmark. Designed for Kaggle T4 / P100 runs.",
    )
    p.add_argument(
        "--include-training",
        action="store_true",
        help="Add the real-NN training sweep (MNIST + CIFAR). Slow on CPU; "
        "implied by --all.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Replace each sweep with its smoke-test variant. Whole script "
        "finishes in ~5 minutes; useful to verify the pipeline before "
        "kicking off the real overnight run.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If one sweep fails, keep going. Default is to stop.",
    )
    args = p.parse_args()

    if args.quick:
        # Tiny grid for every sweep so the orchestrator finishes fast.
        sweeps = ["bench-synthetic-quick"]
        if args.include_gpu or args.all:
            sweeps.append("bench-gpu-quick")
        if args.all or args.include_training:
            sweeps.append("bench-training-quick")
    elif args.all:
        # The full standardised Kaggle suite.
        sweeps = [
            "ablation-init",
            "ablation-swarm",
            "bench-hpo",
            "bench-synthetic",
            "bench-gpu",
            "bench-training",
        ]
    else:
        sweeps = args.sweeps or [
            "ablation-init",
            "ablation-swarm",
            "bench-hpo",
            "bench-synthetic",
        ]
        if args.include_gpu:
            sweeps.append("bench-gpu")
        if args.include_training:
            sweeps.append("bench-training")

    # Print the hardware fingerprint *once*, loudly, at the very top --
    # so when results land in a PR or issue, the machine that produced
    # them is unambiguous.
    from swarmtorch.benchmark.hardware import print_banner

    print_banner(extra={"sweeps": ", ".join(sweeps)})
    print(f"[cloud_bench] python={sys.executable}")
    print(f"[cloud_bench] cwd={REPO_ROOT}")

    failures: list[str] = []
    for name in sweeps:
        ok = _run(name, SWEEPS[name], args.dry_run)
        if not ok:
            failures.append(name)
            if not args.continue_on_error:
                print(
                    f"\n[cloud_bench] STOPPING after {name} failed. "
                    f"Use --continue-on-error to keep going."
                )
                return 1

    print("\n========== SUMMARY ==========")
    print(f"Sweeps run     : {len(sweeps)}")
    print(f"Failures       : {len(failures)}")
    if failures:
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nAll sweeps complete. Results under: results/")
    print("Next: commit results/ to a branch (or use a Jules PR workflow).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

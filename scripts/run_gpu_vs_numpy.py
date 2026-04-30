"""GPU vs NumPy wall-clock comparison sweep.

Validates the headline claim of the paper: ``swarmtorch`` running PSO on
GPU is faster than the equivalent pure-NumPy implementation, especially
as ``swarm_size`` and ``dim`` grow.

Designed to run unchanged on Colab and Kaggle. Switch the runtime to a
T4 / V100 / A100 and pass ``--gpu`` (the default).

Usage:
    python scripts/run_gpu_vs_numpy.py
    python scripts/run_gpu_vs_numpy.py --no-gpu       # CPU-only (sanity check)
    python scripts/run_gpu_vs_numpy.py --quick        # Tiny grid, ~30s
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarmtorch.benchmark.gpu_vs_numpy import run_gpu_vs_numpy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/gpu_vs_numpy"))
    p.add_argument(
        "--functions",
        nargs="+",
        default=["sphere", "rastrigin", "ackley"],
        choices=["sphere", "rastrigin", "rosenbrock", "ackley", "griewank"],
    )
    p.add_argument("--dims", type=int, nargs="+", default=[100, 1000])
    p.add_argument("--swarm-sizes", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--max-fe", type=int, default=20000)
    p.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip the CUDA variant (use on CPU-only machines).",
    )
    p.add_argument(
        "--no-cpu-loop",
        action="store_true",
        help="Skip the legacy per-particle Python loop variant.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Tiny grid for smoke-testing; runs in ~30 seconds.",
    )
    args = p.parse_args()

    if args.quick:
        args.functions = ["sphere", "rastrigin"]
        args.dims = [50]
        args.swarm_sizes = [32, 128]
        args.seeds = [0]
        args.max_fe = 2000

    print(
        f"[run_gpu_vs_numpy] grid: {len(args.functions)} funcs x "
        f"{len(args.dims)} dims x {len(args.swarm_sizes)} swarms x "
        f"{len(args.seeds)} seeds, max_fe={args.max_fe}"
    )

    run_gpu_vs_numpy(
        functions=args.functions,
        dims=args.dims,
        swarm_sizes=args.swarm_sizes,
        seeds=args.seeds,
        max_fe=args.max_fe,
        output_dir=args.output_dir,
        include_cuda=not args.no_gpu,
        include_cpu_loop=not args.no_cpu_loop,
    )

    print(f"\nReport: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

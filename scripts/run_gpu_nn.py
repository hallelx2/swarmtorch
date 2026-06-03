"""Multi-algorithm GPU speedup on a real NN objective (run on Kaggle/Colab GPU).

Usage (Kaggle GPU runtime):
    python scripts/run_gpu_nn.py --output-dir results/gpu_nn

CPU-only sanity check (no cuda rows):
    python scripts/run_gpu_nn.py --no-gpu --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarmtorch.benchmark.gpu_nn import run_nn_speedup
from swarmtorch.benchmark.hardware import print_banner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/gpu_nn"))
    p.add_argument("--algorithms", nargs="+",
                   default=["PSO", "GWO", "WOA", "DE", "CMAES"])
    p.add_argument("--hiddens", type=int, nargs="+", default=[64, 256, 512])
    p.add_argument("--swarm-sizes", type=int, nargs="+", default=[64, 256])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--max-fe", type=int, default=6000)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--no-cpu-loop", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.algorithms = ["PSO", "GWO"]
        args.hiddens = [64]
        args.swarm_sizes = [64]
        args.seeds = [0]
        args.max_fe = 512

    print_banner(extra={"benchmark": "gpu_nn multi-algorithm real-NN speedup"})
    run_nn_speedup(
        hiddens=args.hiddens, algorithms=args.algorithms,
        swarm_sizes=args.swarm_sizes, seeds=args.seeds, max_fe=args.max_fe,
        output_dir=args.output_dir, include_cuda=not args.no_gpu,
        include_cpu_loop=not args.no_cpu_loop,
    )
    print(f"\nReport: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

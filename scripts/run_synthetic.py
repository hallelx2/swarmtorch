"""Stage 4.1 — Synthetic optimization-suite sweep.

Run the curated paper algorithms on the classical black-box test
functions across the dimensionality ladder, with multi-seed replication.

Usage:
    python scripts/run_synthetic.py \
        --output-dir results/synthetic \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --max-fe 5000 \
        --dims 10 50 200

Setting ``--dims 10 50 200 1000`` reproduces the dimensionality-wall
study used in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarmtorch.benchmark import BenchmarkConfig
from swarmtorch.experiments import (
    PAPER_ALGORITHMS,
    SYNTHETIC_FUNCTIONS,
    make_synthetic_tasks,
    run_sweep,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--max-fe", type=int, default=5000)
    p.add_argument("--swarm-size", type=int, default=30)
    p.add_argument(
        "--functions",
        nargs="+",
        default=list(SYNTHETIC_FUNCTIONS),
        choices=list(SYNTHETIC_FUNCTIONS),
    )
    p.add_argument("--dims", type=int, nargs="+", default=[10, 50, 200])
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=list(PAPER_ALGORITHMS),
    )
    args = p.parse_args()

    tasks = make_synthetic_tasks(func_names=args.functions, dims=args.dims)
    config = BenchmarkConfig(
        seeds=args.seeds,
        max_fe=args.max_fe,
        log_every=max(args.max_fe // 50, 1),
        output_dir=args.output_dir,
    )
    report = run_sweep(
        tasks=tasks,
        algorithm_names=args.algorithms,
        config=config,
        swarm_size=args.swarm_size,
        task_kind="synthetic",
    )
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()

"""Stage 4.4 — Ablation studies.

Three ablations are built in:

* ``--ablation init_strategy`` — model vs uniform vs gaussian initialization.
  This is the falsifiable test of the Stage 1.1 fix.
* ``--ablation swarm_size`` — sweep swarm_size in {10, 30, 100, 300}.
* ``--ablation max_fe`` — three FE budgets {1k, 10k, 100k}.

Run with:

    python scripts/run_ablations.py --ablation init_strategy \
        --output-dir results/ablations/init \
        --algorithms PSO CA TLBO --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swarmtorch.benchmark import BenchmarkConfig
from swarmtorch.experiments import run_sweep
from swarmtorch.experiments.synthetic import make_synthetic_tasks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ablation",
        choices=["init_strategy", "swarm_size", "max_fe"],
        required=True,
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--max-fe", type=int, default=3000)
    p.add_argument("--swarm-size", type=int, default=30)
    p.add_argument(
        "--algorithms",
        nargs="+",
        default=["PSO", "CA", "TLBO", "CMAES"],
        help="Subset of PAPER_ALGORITHMS to ablate.",
    )
    p.add_argument(
        "--functions", nargs="+", default=["rastrigin", "rosenbrock"],
    )
    p.add_argument("--dims", type=int, nargs="+", default=[50])
    args = p.parse_args()

    tasks = make_synthetic_tasks(func_names=args.functions, dims=args.dims)
    config = BenchmarkConfig(
        seeds=args.seeds,
        max_fe=args.max_fe,
        log_every=max(args.max_fe // 50, 1),
        output_dir=args.output_dir,
    )

    if args.ablation == "init_strategy":
        axes = {"init_strategy": ["model", "uniform", "gaussian"]}
    elif args.ablation == "swarm_size":
        axes = {"swarm_size": [10, 30, 100, 300]}
    else:  # max_fe is handled by re-running config.max_fe externally; documented but not auto-axes
        raise SystemExit(
            "max_fe ablation: re-run this script three times with "
            "--max-fe 1000, --max-fe 10000, --max-fe 100000 and merge "
            "the output dirs."
        )

    report = run_sweep(
        tasks=tasks,
        algorithm_names=args.algorithms,
        config=config,
        swarm_size=args.swarm_size,
        ablation_axes=axes,
        task_kind="synthetic",
    )
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()

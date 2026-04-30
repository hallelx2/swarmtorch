"""Stage 4.3 — Real HPO sweep.

Runs the curated metaheuristic searchers + Random / TPE / Hyperband
baselines on the three reference HPO tasks (small CNN, tiny
transformer, optionally XGBoost-tabular).

Each searcher gets the same trial budget; results land in JSONs that
the report builder can aggregate.

Usage:
    python scripts/run_hpo.py \
        --output-dir results/hpo \
        --seeds 0 1 2 3 4 \
        --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from swarmtorch.baselines.hpo import (
    HyperbandSearchBaseline,
    RandomSearchBaseline,
    TPESearchBaseline,
)
from swarmtorch.benchmark.run import RunResult, seed_everything
from swarmtorch.benchmark.report import build_report
from swarmtorch.experiments.hpo import (
    make_cnn_hpo_task,
    make_tiny_transformer_hpo_task,
)


def _search_and_record(
    searcher_cls: Any,
    task: Any,
    seed: int,
    output_dir: Path,
    extra_kwargs: dict | None = None,
) -> None:
    seed_everything(seed)
    extra_kwargs = extra_kwargs or {}
    s = searcher_cls(
        model_fn=task.model_fn,
        param_space=task.param_space,
        train_fn=task.train_fn,
        n_trials=task.n_trials,
        device="cpu",
        verbose=False,
        seed=seed,
        **extra_kwargs,
    )
    t0 = time.perf_counter()
    result = s.search()
    wall = time.perf_counter() - t0

    rr = RunResult(
        algo_name=searcher_cls.__name__,
        task_name=task.name,
        seed=seed,
        final_score=float(result.best_score),
        wall_seconds=float(wall),
        peak_mem_mb=0.0,
        fe_used=len(result.history),
        trajectory=[(i + 1, score) for i, (_, score, _) in enumerate(result.history)],
        meta={"best_params": result.best_params},
    )
    rr.save(output_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/hpo"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--with-xgboost", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        make_cnn_hpo_task(),
        make_tiny_transformer_hpo_task(),
    ]
    if args.with_xgboost:
        from swarmtorch.experiments.hpo import make_xgboost_hpo_task

        tasks.append(make_xgboost_hpo_task())
    for t in tasks:
        t.n_trials = args.n_trials

    searchers = [RandomSearchBaseline, TPESearchBaseline, HyperbandSearchBaseline]
    cell_idx = 0
    n_cells = len(tasks) * len(searchers) * len(args.seeds)
    for task in tasks:
        for searcher_cls in searchers:
            for seed in args.seeds:
                cell_idx += 1
                print(
                    f"[run_hpo] cell {cell_idx}/{n_cells}: task={task.name} "
                    f"searcher={searcher_cls.__name__} seed={seed}"
                )
                _search_and_record(searcher_cls, task, seed, args.output_dir)

    report = build_report(args.output_dir, title="swarmtorch HPO sweep")
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()

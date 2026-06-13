"""Stage 4.3 — Real HPO sweep.

Compares swarmtorch's metaheuristic HPO searchers against the
Random / TPE / Hyperband baselines on the reference HPO tasks
(small CNN, tiny transformer, optionally XGBoost-tabular).

Fairness: every method gets the same evaluation budget, expressed as
the number of hyperparameter configurations trained (``--n-trials``).
The Optuna/Random baselines run ``n_trials`` trials directly; the
population-based searchers run ``iterations x swarm_size = n_trials``
model trainings, so no method gets to evaluate more configurations
than another.

Usage:
    python scripts/run_hpo.py \
        --output-dir results/hpo \
        --seeds 0 1 2 3 4 \
        --n-trials 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from swarmtorch.baselines.hpo import (
    HyperbandSearchBaseline,
    RandomSearchBaseline,
    TPESearchBaseline,
)
from swarmtorch.benchmark.report import build_report
from swarmtorch.benchmark.run import RunResult, seed_everything
from swarmtorch.experiments.hpo import (
    make_cnn_hpo_task,
    make_tiny_transformer_hpo_task,
)

# Metaheuristic HPO searchers (curated paper short-list).
from swarmtorch import (
    CASearch,
    CMAESSearch,
    FPASearch,
    GorillaSearch,
    PSOSearch,
    SineCosineSearch,
    TLBOSearch,
)

# Population size for the metaheuristic searchers. HPO spaces here are
# low-dimensional (<= 4 hyperparameters), so a small swarm is appropriate;
# iterations are derived to hold the total evaluation budget equal to the
# baselines' n_trials.
META_SWARM_SIZE = 5

BASELINE_SEARCHERS = {
    "RandomSearch": RandomSearchBaseline,
    "TPE": TPESearchBaseline,
    "Hyperband": HyperbandSearchBaseline,
}
META_SEARCHERS = {
    "PSOSearch": PSOSearch,
    "CMAESSearch": CMAESSearch,
    "CASearch": CASearch,
    "TLBOSearch": TLBOSearch,
    "FPASearch": FPASearch,
    "GorillaSearch": GorillaSearch,
    "SineCosineSearch": SineCosineSearch,
}


def _record(
    algo_name: str,
    task: Any,
    seed: int,
    best_score: float,
    best_params: dict,
    wall: float,
    fe_used: int,
    output_dir: Path,
) -> None:
    RunResult(
        algo_name=algo_name,
        task_name=task.name,
        seed=seed,
        final_score=float(best_score),
        wall_seconds=float(wall),
        peak_mem_mb=0.0,
        fe_used=int(fe_used),
        trajectory=[],
        meta={"best_params": best_params},
    ).save(output_dir)


def _run_baseline(name, cls, task, seed, n_trials, output_dir):
    seed_everything(seed)
    s = cls(
        model_fn=task.model_fn,
        param_space=task.param_space,
        train_fn=task.train_fn,
        n_trials=n_trials,
        device="cpu",
        verbose=False,
        seed=seed,
    )
    t0 = time.perf_counter()
    result = s.search()
    wall = time.perf_counter() - t0
    _record(
        name, task, seed, result.best_score, result.best_params, wall,
        len(result.history), output_dir,
    )


def _run_meta(name, cls, task, seed, n_trials, output_dir):
    seed_everything(seed)
    iterations = max(1, n_trials // META_SWARM_SIZE)
    s = cls(
        model_fn=task.model_fn,
        param_space=task.param_space,
        train_fn=task.train_fn,
        iterations=iterations,
        swarm_size=META_SWARM_SIZE,
        device="cpu",
        verbose=False,
    )
    t0 = time.perf_counter()
    best_params = s.search()
    wall = time.perf_counter() - t0
    best_score = s.best_score if s.best_score is not None else float("inf")
    _record(
        name, task, seed, best_score, best_params, wall,
        iterations * META_SWARM_SIZE, output_dir,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/hpo"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--with-xgboost", action="store_true")
    p.add_argument(
        "--baselines-only",
        action="store_true",
        help="Skip the metaheuristic searchers (Optuna baselines only).",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [make_cnn_hpo_task(), make_tiny_transformer_hpo_task()]
    if args.with_xgboost:
        from swarmtorch.experiments.hpo import make_xgboost_hpo_task

        tasks.append(make_xgboost_hpo_task())
    for t in tasks:
        t.n_trials = args.n_trials

    searchers: list[tuple[str, Any, bool]] = [
        (name, cls, True) for name, cls in BASELINE_SEARCHERS.items()
    ]
    if not args.baselines_only:
        searchers += [
            (name, cls, False) for name, cls in META_SEARCHERS.items()
        ]

    n_cells = len(tasks) * len(searchers) * len(args.seeds)
    cell = 0
    for task in tasks:
        for name, cls, is_baseline in searchers:
            for seed in args.seeds:
                cell += 1
                print(
                    f"[run_hpo] cell {cell}/{n_cells}: task={task.name} "
                    f"searcher={name} seed={seed}",
                    flush=True,
                )
                try:
                    if is_baseline:
                        _run_baseline(name, cls, task, seed, args.n_trials, args.output_dir)
                    else:
                        _run_meta(name, cls, task, seed, args.n_trials, args.output_dir)
                except Exception as e:  # keep the sweep alive on a single failure
                    print(f"  !! {name} failed on {task.name} seed {seed}: {e}")

    report = build_report(args.output_dir, title="swarmtorch HPO sweep")
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()

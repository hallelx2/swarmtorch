"""Stage 4.3b — HPO on real-signal sklearn datasets.

Same comparison as run_hpo.py (metaheuristic searchers vs Random / TPE /
Hyperband, equal evaluation budget) but on five real datasets where good
hyperparameters genuinely matter -- so the searchers can actually
separate and the Friedman test has power (N=5 tasks).

Usage:
    python scripts/run_hpo_real.py --output-dir results/hpo_real \
        --seeds 0 1 2 3 4 --n-trials 25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from swarmtorch.baselines.hpo import (
    HyperbandSearchBaseline,
    RandomSearchBaseline,
    TPESearchBaseline,
)
from swarmtorch.benchmark.report import build_report
from swarmtorch.benchmark.run import RunResult, seed_everything
from swarmtorch.experiments.hpo_real import make_real_hpo_tasks
from swarmtorch import (
    CASearch,
    CMAESSearch,
    FPASearch,
    GorillaSearch,
    PSOSearch,
    SineCosineSearch,
    TLBOSearch,
)

META_SWARM_SIZE = 5
BASELINES = {"RandomSearch": RandomSearchBaseline, "TPE": TPESearchBaseline,
             "Hyperband": HyperbandSearchBaseline}
META = {"PSOSearch": PSOSearch, "CMAESSearch": CMAESSearch, "CASearch": CASearch,
        "TLBOSearch": TLBOSearch, "FPASearch": FPASearch,
        "GorillaSearch": GorillaSearch, "SineCosineSearch": SineCosineSearch}


def _record(name, task, seed, score, params, wall, fe, out):
    RunResult(
        algo_name=name, task_name=task.name, seed=seed, final_score=float(score),
        wall_seconds=float(wall), peak_mem_mb=0.0, fe_used=int(fe),
        trajectory=[], meta={"best_params": params},
    ).save(out)


def _run_baseline(name, cls, task, seed, n_trials, out):
    seed_everything(seed)
    s = cls(model_fn=task.model_fn, param_space=task.param_space, train_fn=task.train_fn,
            n_trials=n_trials, device="cpu", verbose=False, seed=seed)
    t0 = time.perf_counter(); r = s.search(); wall = time.perf_counter() - t0
    _record(name, task, seed, r.best_score, r.best_params, wall, len(r.history), out)


def _run_meta(name, cls, task, seed, n_trials, out):
    seed_everything(seed)
    iters = max(1, n_trials // META_SWARM_SIZE)
    s = cls(model_fn=task.model_fn, param_space=task.param_space, train_fn=task.train_fn,
            iterations=iters, swarm_size=META_SWARM_SIZE, device="cpu", verbose=False)
    t0 = time.perf_counter(); best = s.search(); wall = time.perf_counter() - t0
    score = s.best_score if s.best_score is not None else float("inf")
    _record(name, task, seed, score, best, wall, iters * META_SWARM_SIZE, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/hpo_real"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--baselines-only", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = make_real_hpo_tasks()
    for t in tasks:
        t.n_trials = args.n_trials

    searchers = [(n, c, True) for n, c in BASELINES.items()]
    if not args.baselines_only:
        searchers += [(n, c, False) for n, c in META.items()]

    n_cells = len(tasks) * len(searchers) * len(args.seeds)
    cell = 0
    for task in tasks:
        for name, cls, is_base in searchers:
            for seed in args.seeds:
                cell += 1
                print(f"[run_hpo_real] {cell}/{n_cells}: task={task.name} "
                      f"searcher={name} seed={seed}", flush=True)
                try:
                    if is_base:
                        _run_baseline(name, cls, task, seed, args.n_trials, args.output_dir)
                    else:
                        _run_meta(name, cls, task, seed, args.n_trials, args.output_dir)
                except Exception as e:
                    print(f"  !! {name} failed on {task.name} seed {seed}: {e}")

    report = build_report(args.output_dir, title="swarmtorch HPO sweep (real datasets)")
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()

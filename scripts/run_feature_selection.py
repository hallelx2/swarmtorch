"""Stage 4.6 — Combinatorial feature selection (gradient-free application).

Compares swarmtorch metaheuristics, a random-selection baseline, and the
external Nevergrad library on selecting K of N features. The objective is
validation error of a logistic-regression classifier on the selected
features -- piecewise-constant, no gradient. Every method gets the same
function-evaluation budget.

Usage:
    python scripts/run_feature_selection.py --output-dir results/feature_selection \
        --seeds 0 1 2 3 4 5 6 7 8 9 --max-fe 600
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swarmtorch import PSO, GWO, DE, WOA
from swarmtorch.evolutionary.model_training import CMAES
from swarmtorch.benchmark.budget import FEBudgetTracker
from swarmtorch.benchmark.report import build_report
from swarmtorch.benchmark.run import RunResult, seed_everything
from swarmtorch.experiments.feature_selection import make_feature_selection_problem

SWARM_ALGOS = {"PSO": PSO, "GWO": GWO, "DE": DE, "WOA": WOA, "CMAES": CMAES}


def _record(name, task_name, seed, score, wall, fe, out, meta=None):
    RunResult(
        algo_name=name, task_name=task_name, seed=seed, final_score=float(score),
        wall_seconds=float(wall), peak_mem_mb=0.0, fe_used=int(fe),
        trajectory=[], meta=meta or {},
    ).save(out)


def run_swarm(name, cls, prob, seed, max_fe, swarm_size, out):
    seed_everything(seed)
    module = prob.make_module()
    kwargs = {"swarm_size": swarm_size}
    # CMA-ES doesn't take init_strategy in older signature; PSO etc. do.
    try:
        opt = cls(module.parameters(), init_strategy="uniform", **kwargs)
    except TypeError:
        opt = cls(module.parameters(), **kwargs)
    tracker = FEBudgetTracker(max_fe=max_fe)
    closure = tracker.wrap_closure(lambda: torch.tensor(prob.evaluate(module.x)))
    t0 = time.perf_counter()
    while not tracker.done:
        opt.step(closure)
    wall = time.perf_counter() - t0
    _record(name, prob.name, seed, tracker.best_so_far, wall, tracker.fe_count, out)


def run_random(prob, seed, max_fe, out):
    rng = np.random.default_rng(seed)
    best = float("inf")
    t0 = time.perf_counter()
    for _ in range(max_fe):
        scores = torch.tensor(rng.random(prob.n_features))
        e = prob.evaluate(scores)
        best = min(best, e)
    wall = time.perf_counter() - t0
    _record("RandomSelection", prob.name, seed, best, wall, max_fe, out)


def run_nevergrad(opt_name, prob, seed, max_fe, out):
    import nevergrad as ng

    param = ng.p.Array(shape=(prob.n_features,)).set_bounds(0.0, 1.0)
    optimizer = ng.optimizers.registry[opt_name](
        parametrization=param, budget=max_fe, num_workers=1
    )
    optimizer.parametrization.random_state.seed(seed)
    best = float("inf")
    t0 = time.perf_counter()
    for _ in range(max_fe):
        cand = optimizer.ask()
        e = prob.evaluate(torch.tensor(np.asarray(cand.value)))
        optimizer.tell(cand, e)
        best = min(best, e)
    wall = time.perf_counter() - t0
    _record(f"Nevergrad-{opt_name}", prob.name, seed, best, wall, max_fe, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("results/feature_selection"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--max-fe", type=int, default=600)
    p.add_argument("--swarm-size", type=int, default=20)
    p.add_argument("--n-features", type=int, default=50)
    p.add_argument("--n-informative", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(SWARM_ALGOS) + ["RandomSelection", "Nevergrad-PSO", "Nevergrad-CMA"]
    n_cells = len(methods) * len(args.seeds)
    cell = 0
    for seed in args.seeds:
        prob = make_feature_selection_problem(
            n_features=args.n_features, n_informative=args.n_informative,
            k=args.k, seed=seed,
        )
        for name in methods:
            cell += 1
            print(f"[featsel] {cell}/{n_cells}: method={name} seed={seed}", flush=True)
            try:
                if name in SWARM_ALGOS:
                    run_swarm(name, SWARM_ALGOS[name], prob, seed, args.max_fe,
                              args.swarm_size, args.output_dir)
                elif name == "RandomSelection":
                    run_random(prob, seed, args.max_fe, args.output_dir)
                elif name == "Nevergrad-PSO":
                    run_nevergrad("PSO", prob, seed, args.max_fe, args.output_dir)
                elif name == "Nevergrad-CMA":
                    run_nevergrad("CMA", prob, seed, args.max_fe, args.output_dir)
            except Exception as e:
                print(f"  !! {name} failed seed {seed}: {e}")

    report = build_report(args.output_dir, title="swarmtorch feature selection")
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    main()

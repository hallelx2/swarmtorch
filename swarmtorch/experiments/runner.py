"""Sweep runner for Stage 4 paper experiments.

A sweep is a Cartesian product of ``(task, algorithm, seed)`` triples.
The runner instantiates each cell from scratch (so seeding takes
effect), wires the task's closure through an
:class:`FEBudgetTracker`, drives the optimizer until budget exhaustion,
and persists per-cell ``RunResult`` JSONs. After every cell completes,
:func:`run_sweep` builds the markdown report and the per-task
convergence plots.

For Stage 4.4 (ablations), pass an ``ablation_axes`` dict — e.g.
``{"init_strategy": ["model", "uniform", "gaussian"]}`` — to product
across additional dimensions; each axis value becomes part of the
algorithm name in the report (so ``PSO`` becomes ``PSO[init=model]``,
``PSO[init=uniform]``, etc.) and shows up as separate columns in the
results table and convergence plot.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import torch

from swarmtorch.benchmark import (
    BenchmarkConfig,
    build_report,
    convergence_plot,
    load_results,
    run_one,
)
from swarmtorch.benchmark.budget import FEBudgetTracker
from swarmtorch.experiments.registry import build_algorithm_factory


class TaskAdapter(Protocol):
    """Common interface tasks must satisfy to be sweep-runnable.

    Both :class:`SyntheticTask` (synthetic test functions) and
    :class:`TrainingTask` (real NN training) satisfy this protocol via
    duck-typing — they each provide ``name``, ``make_module()``-or-equivalent,
    and a way to build a closure.
    """

    name: str

    def build_closure(
        self, optimizer_constructor_input: Any
    ) -> tuple[Any, Callable[[], torch.Tensor]]: ...


def _build_synthetic_runner(
    task: Any, algo_name: str, swarm_size: int, init_strategy: str, max_fe: int
) -> Callable[[], tuple[Any, Callable[[], float]]]:
    """Build an ``algo_factory`` for a SyntheticTask + algorithm.

    The closure is wrapped through the FEBudgetTracker so the runner can
    enforce a fair function-evaluation budget across heterogeneous
    optimizers.
    """
    factory = build_algorithm_factory(
        algo_name,
        swarm_size=swarm_size,
        init_strategy=init_strategy,
    )

    def algo_factory() -> tuple[Any, Callable[[], float]]:
        module = task.make_module()
        optimizer = factory(module)
        plain = task.make_closure(module)

        # Tracker is owned by the runner; we wire the closure through it.
        tracker = FEBudgetTracker(max_fe=max_fe, log_every=max(max_fe // 50, 1))

        # Attach so the outer driver can read tracker state if needed.
        optimizer._fe_tracker = tracker  # type: ignore[attr-defined]

        wrapped_closure = tracker.wrap_closure(plain)

        def step_fn() -> float:
            optimizer.step(wrapped_closure)
            return float(tracker.best_so_far)

        return optimizer, step_fn

    return algo_factory


def _build_training_runner(
    task: Any,
    algo_name: str,
    swarm_size: int,
    init_strategy: str,
    max_fe: int,
) -> Callable[[], tuple[Any, Callable[[], float]]]:
    """Build an ``algo_factory`` for a TrainingTask + algorithm.

    Mini-batches roll forward between optimizer steps; FE budget is in
    *closure invocations* (one per particle evaluation, one per
    minibatch for gradient methods).
    """
    factory = build_algorithm_factory(
        algo_name,
        swarm_size=swarm_size,
        init_strategy=init_strategy,
    )

    def algo_factory() -> tuple[Any, Callable[[], float]]:
        model = task.model_factory()
        train_loader, _ = task.make_loaders()
        plain = task.make_closure(model, train_loader)
        optimizer = factory(model)

        tracker = FEBudgetTracker(max_fe=max_fe, log_every=max(max_fe // 50, 1))
        optimizer._fe_tracker = tracker  # type: ignore[attr-defined]
        wrapped_closure = tracker.wrap_closure(plain)

        def step_fn() -> float:
            optimizer.step(wrapped_closure)
            # Roll to the next minibatch so the FE budget genuinely
            # samples different data points over the run.
            advance = getattr(plain, "advance_batch", None)
            if advance is not None:
                advance()
            return float(tracker.best_so_far)

        return optimizer, step_fn

    return algo_factory


def _label(name: str, axes: dict[str, Any] | None) -> str:
    if not axes:
        return name
    suffix = ",".join(f"{k}={v}" for k, v in axes.items())
    return f"{name}[{suffix}]"


def run_sweep(
    tasks: list[Any],
    algorithm_names: list[str],
    config: BenchmarkConfig,
    swarm_size: int = 30,
    init_strategy: str = "model",
    ablation_axes: dict[str, list[Any]] | None = None,
    task_kind: str = "synthetic",
    build_plots: bool = True,
) -> Path:
    """Run a Cartesian sweep and produce the final report.

    Args:
        tasks: List of ``SyntheticTask`` or ``TrainingTask`` instances.
        algorithm_names: Names from
            :data:`swarmtorch.experiments.registry.PAPER_ALGORITHMS`.
        config: Shared run settings (seeds, max_fe, output_dir).
        swarm_size: Default swarm size; overridden by an ablation axis if
            ``ablation_axes`` contains ``"swarm_size"``.
        init_strategy: Default init strategy; overridden by
            ``ablation_axes`` if it contains ``"init_strategy"``.
        ablation_axes: Optional extra dimensions to product over. Each
            axis value becomes part of the algorithm label in the report.
        task_kind: ``"synthetic"`` or ``"training"`` — picks the
            appropriate runner builder.
        build_plots: Generate per-task convergence PNGs after the sweep.

    Returns:
        Path to the markdown report.
    """
    if task_kind not in {"synthetic", "training"}:
        raise ValueError(f"task_kind must be 'synthetic' or 'training', got {task_kind!r}")

    builder = (
        _build_synthetic_runner if task_kind == "synthetic" else _build_training_runner
    )

    # Materialize the cartesian product over ablation axes.
    axes = ablation_axes or {}
    axis_keys = list(axes.keys())
    axis_values = [axes[k] for k in axis_keys]
    axis_combinations: list[dict[str, Any]] = (
        [
            dict(zip(axis_keys, combo))
            for combo in itertools.product(*axis_values)
        ]
        if axis_values
        else [{}]
    )

    n_cells = len(tasks) * len(algorithm_names) * len(axis_combinations) * len(config.seeds)
    print(f"[run_sweep] {n_cells} cells: {len(tasks)} tasks x {len(algorithm_names)} algos "
          f"x {len(axis_combinations)} axis combos x {len(config.seeds)} seeds")

    cell_idx = 0
    for task in tasks:
        for algo_name in algorithm_names:
            for axes_assignment in axis_combinations:
                effective_swarm = int(axes_assignment.get("swarm_size", swarm_size))
                effective_init = str(axes_assignment.get("init_strategy", init_strategy))
                label = _label(algo_name, axes_assignment)
                for seed in config.seeds:
                    cell_idx += 1
                    print(
                        f"[run_sweep] cell {cell_idx}/{n_cells}: "
                        f"task={task.name} algo={label} seed={seed}"
                    )
                    af = builder(
                        task,
                        algo_name,
                        swarm_size=effective_swarm,
                        init_strategy=effective_init,
                        max_fe=config.max_fe,
                    )
                    cell_meta: dict[str, Any] = {
                        "swarm_size": effective_swarm,
                        "init_strategy": effective_init,
                    }
                    cell_meta.update(axes_assignment)
                    run_one(
                        algo_factory=af,
                        task_name=task.name,
                        algo_name=label,
                        config=config,
                        seed=seed,
                        meta=cell_meta,
                    )

    # Aggregate.
    report_path = build_report(
        config.output_dir,
        title="swarmtorch sweep results",
    )

    if build_plots:
        all_results = load_results(config.output_dir)
        for task in tasks:
            task_results = [r for r in all_results if r.task_name == task.name]
            if not task_results:
                continue
            convergence_plot(
                task_results,
                config.output_dir / f"convergence_{task.name}.png",
                title=task.name,
            )

    return report_path

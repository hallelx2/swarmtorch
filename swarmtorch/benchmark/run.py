"""Multi-seed benchmark runner with FE budgets and compute reporting.

The runner is deliberately small: every algorithm we can benchmark
already exposes a ``step(closure)`` method and (optionally)
``set_functional_closure`` for the vmap fast path. The runner adapts
those into a uniform driver that:

1. Seeds Python / NumPy / PyTorch deterministically.
2. Wraps the task's closure in an :class:`FEBudgetTracker` so every
   algorithm spends the same number of function evaluations.
3. Calls ``step()`` until the FE budget is exhausted, recording wall
   clock and (when available) peak GPU memory.
4. Persists results as JSON, one file per (algorithm, task, seed) cell.

Stage 4 will use :func:`run_one` from a sweep loop; Stage 5 reads the
resulting JSONs to build the report.
"""

from __future__ import annotations

import json
import random
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from swarmtorch.benchmark.budget import FEBudgetTracker


def seed_everything(seed: int) -> None:
    """Seed every RNG that matters for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level knobs shared across every cell in a sweep."""

    seeds: list[int]
    max_fe: int
    log_every: int = 50
    output_dir: Path = Path("results")
    device: str = "cpu"

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class RunResult:
    """One (algorithm, task, seed) cell of the benchmark grid."""

    algo_name: str
    task_name: str
    seed: int
    final_score: float
    wall_seconds: float
    peak_mem_mb: float
    fe_used: int
    trajectory: list[tuple[int, float]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Path) -> Path:
        path = (
            Path(output_dir)
            / f"{self.task_name}__{self.algo_name}__seed{self.seed}.json"
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=_jsonable)
        return path

    @classmethod
    def load(cls, path: Path) -> "RunResult":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Trajectories serialize as list[list[int, float]] — restore tuples.
        data["trajectory"] = [tuple(p) for p in data.get("trajectory", [])]
        return cls(**data)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (Path, np.integer, np.floating)):
        return str(obj) if isinstance(obj, Path) else obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class StepFn(Protocol):
    """Driver-side contract for one optimizer step.

    Returns the current best loss as a float (used as a fallback when the
    tracker can't observe the value, e.g. for gradient baselines that
    don't return a tensor).
    """

    def __call__(self) -> float: ...


def run_one(
    algo_factory: Callable[[], tuple[Any, StepFn]],
    task_name: str,
    algo_name: str,
    config: BenchmarkConfig,
    seed: int,
    score_fn: Callable[[Any], float] | None = None,
    meta: dict[str, Any] | None = None,
) -> RunResult:
    """Run a single (algorithm, task, seed) cell.

    Args:
        algo_factory: Zero-argument callable returning ``(optimizer, step_fn)``.
            ``step_fn()`` does one optimizer step and returns a float
            "current best loss". The factory must build the optimizer from
            scratch so seeding takes effect.
        task_name: Identifier persisted in the result JSON.
        algo_name: Identifier persisted in the result JSON.
        config: Shared benchmark settings.
        seed: This run's seed.
        score_fn: Optional callable receiving the optimizer and returning
            the final score (e.g. a clean test-set evaluation). If None,
            the last value from ``step_fn`` is used.
        meta: Free-form dict persisted alongside the result.
    """
    seed_everything(seed)

    tracker = FEBudgetTracker(max_fe=config.max_fe, log_every=config.log_every)
    optimizer, step_fn = algo_factory()

    # If the user's algo_factory wired up a closure that goes through the
    # tracker (recommended), we trust the tracker. Otherwise, the runner
    # advances the FE counter conservatively from step_fn's return value.
    use_tracemalloc = config.device == "cpu"
    if use_tracemalloc:
        tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    last_score = float("inf")
    while not tracker.done:
        before = tracker.fe_count
        last_score = float(step_fn())
        # If step_fn didn't advance the tracker (gradient baseline path),
        # bump it by 1 so the loop terminates.
        if tracker.fe_count == before:
            tracker.observe_external(1, last_score)
    wall = time.perf_counter() - t0

    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif use_tracemalloc:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mem_mb = peak / (1024 * 1024)
    else:
        peak_mem_mb = 0.0

    final_score = score_fn(optimizer) if score_fn is not None else last_score

    result = RunResult(
        algo_name=algo_name,
        task_name=task_name,
        seed=seed,
        final_score=float(final_score),
        wall_seconds=float(wall),
        peak_mem_mb=float(peak_mem_mb),
        fe_used=int(tracker.fe_count),
        trajectory=tracker.trajectory,
        meta=meta or {},
    )
    result.save(config.output_dir)
    return result


def load_results(output_dir: Path) -> list[RunResult]:
    """Load every ``RunResult`` JSON in ``output_dir``."""
    return [RunResult.load(p) for p in sorted(Path(output_dir).glob("*.json"))]

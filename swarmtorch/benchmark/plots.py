"""Convergence curves and Demsar critical-difference diagrams.

These two figures are what the paper-grade results look like: every
table-of-numbers in the current ``COMPREHENSIVE_EXPERIMENT_REPORT.md``
should be paired with a convergence plot (showing how each algorithm
gets there) and, when comparing many algorithms across many tasks, a CD
diagram that visualizes which methods are statistically distinguishable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from swarmtorch.benchmark.run import RunResult


def _interpolate_trajectory(
    trajectory: list[tuple[int, float]],
    fe_grid: np.ndarray,
) -> np.ndarray:
    """Step-interpolate a (fe, best) trajectory onto a regular FE grid.

    Trajectories are best-so-far series — they're monotone non-increasing
    in score, so step interpolation is the right choice (linear
    interpolation would invent intermediate values that the algorithm
    never observed).
    """
    if not trajectory:
        return np.full_like(fe_grid, np.nan, dtype=float)
    xs = np.array([t[0] for t in trajectory], dtype=float)
    ys = np.array([t[1] for t in trajectory], dtype=float)
    out = np.empty_like(fe_grid, dtype=float)
    j = 0
    current = np.inf
    for i, fe in enumerate(fe_grid):
        while j < len(xs) and xs[j] <= fe:
            current = ys[j]
            j += 1
        out[i] = current
    return out


def convergence_plot(
    results: Sequence[RunResult],
    output_path: Path,
    title: str | None = None,
    n_grid_points: int = 100,
) -> Path:
    """Mean +/- 1 std convergence curves, one per algorithm.

    Args:
        results: Iterable of ``RunResult`` covering one task with
            multiple algorithms and seeds.
        output_path: Where to write the PNG.
        title: Plot title; defaults to the first result's task name.
        n_grid_points: Number of FE samples for the x-axis.
    """
    import matplotlib.pyplot as plt

    by_algo: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_algo[r.algo_name].append(r)

    if not by_algo:
        raise ValueError("convergence_plot received no results")

    max_fe = max(t[0] for r in results for t in r.trajectory if r.trajectory)
    if max_fe <= 0:
        raise ValueError("no FE-budget trajectories found in results")
    fe_grid = np.linspace(1, max_fe, n_grid_points)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for algo, runs in sorted(by_algo.items()):
        curves = np.array(
            [_interpolate_trajectory(r.trajectory, fe_grid) for r in runs]
        )
        # Use nanmean / nanstd in case some runs ended with no logged steps.
        # Suppress the harmless "invalid value encountered" RuntimeWarning
        # that fires when an entire column is NaN at the start of a run.
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(curves, axis=0)
            std = np.nanstd(curves, axis=0)
        ax.plot(fe_grid, mean, label=algo)
        ax.fill_between(fe_grid, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best loss so far")
    ax.set_title(title or results[0].task_name)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def critical_difference_diagram(
    avg_ranks: Sequence[float],
    names: Sequence[str],
    cd: float,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """Demsar-style critical-difference diagram.

    Algorithms are placed on a horizontal axis at their average rank;
    horizontal bars connect groups whose ranks differ by less than the
    critical difference (i.e. statistically indistinguishable groups).

    Args:
        avg_ranks: Average rank per algorithm (lower-is-better).
        names: Algorithm names, same order as ``avg_ranks``.
        cd: Critical difference value from
            :func:`nemenyi_critical_difference`.
        output_path: Where to write the PNG.
    """
    import matplotlib.pyplot as plt

    if len(avg_ranks) != len(names):
        raise ValueError("avg_ranks and names must have the same length")
    n = len(avg_ranks)
    if n < 2:
        raise ValueError("need at least 2 algorithms")

    order = np.argsort(avg_ranks)
    sorted_ranks = np.asarray(avg_ranks)[order]
    sorted_names = [names[i] for i in order]

    # Group consecutive algorithms whose rank gap < CD.
    groups: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_ranks[j + 1] - sorted_ranks[i] < cd:
            j += 1
        if j > i:
            groups.append((i, j))
        i = j + 1

    fig, ax = plt.subplots(figsize=(8, 2.5 + 0.3 * n), dpi=150)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(-1 - 0.3 * n, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 1))
    ax.set_xlabel(f"Average rank (CD = {cd:.3f})")

    # Tick marks on the rank axis.
    for r in range(1, n + 1):
        ax.axvline(r, ymin=0.95, ymax=1.0, color="black", lw=0.5)

    # Place each algorithm at its rank; alternate labels above and below.
    for k, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        side = -1 if k < n / 2 else 1
        y_text = -0.5 - 0.3 * (k if side < 0 else (n - 1 - k))
        ax.plot([rank, rank], [1, y_text], color="black", lw=0.8)
        ha = "right" if side < 0 else "left"
        text_x = (0.5 if side < 0 else n + 0.5)
        ax.plot([rank, text_x], [y_text, y_text], color="black", lw=0.8)
        ax.text(
            text_x + (-0.05 if side < 0 else 0.05),
            y_text,
            f"{name} ({rank:.2f})",
            ha=ha,
            va="center",
            fontsize=9,
        )

    # Connecting bars for indistinguishable groups.
    for k, (lo, hi) in enumerate(groups):
        ax.plot(
            [sorted_ranks[lo] - 0.02, sorted_ranks[hi] + 0.02],
            [0.5 - 0.05 * k, 0.5 - 0.05 * k],
            color="black",
            lw=2.5,
        )

    if title:
        ax.set_title(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path

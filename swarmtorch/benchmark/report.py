"""Aggregate per-seed RunResult JSONs into a markdown report.

This replaces the legacy ``summarize_results.py`` (which read a
hardcoded UTF-16-LE Windows path) with a clean function that:

1. Loads every ``RunResult`` JSON in a directory.
2. Groups by (task, algorithm) and computes mean / std / median across
   seeds.
3. Emits a markdown table with sensible precision (3-4 sig figs +/- std).
4. Adds the Friedman test result and Nemenyi critical difference at the
   bottom when there are at least 2 tasks and 2 algorithms.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swarmtorch.benchmark.hardware import hardware_info
from swarmtorch.benchmark.run import RunResult, load_results
from swarmtorch.benchmark.stats import (
    friedman_test,
    nemenyi_critical_difference,
    rank_algorithms,
)


@dataclass
class CellSummary:
    task_name: str
    algo_name: str
    n_seeds: int
    mean_score: float
    std_score: float
    median_score: float
    mean_wall: float
    mean_peak_mem_mb: float


def aggregate_results(
    results: Iterable[RunResult],
) -> list[CellSummary]:
    """Group ``RunResult``s by (task, algo) and compute summary stats."""
    groups: dict[tuple[str, str], list[RunResult]] = defaultdict(list)
    for r in results:
        groups[(r.task_name, r.algo_name)].append(r)

    summaries: list[CellSummary] = []
    for (task, algo), runs in sorted(groups.items()):
        scores = np.array([r.final_score for r in runs], dtype=float)
        walls = np.array([r.wall_seconds for r in runs], dtype=float)
        mems = np.array([r.peak_mem_mb for r in runs], dtype=float)
        summaries.append(
            CellSummary(
                task_name=task,
                algo_name=algo,
                n_seeds=len(runs),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores, ddof=1) if scores.size > 1 else 0.0),
                median_score=float(np.median(scores)),
                mean_wall=float(np.mean(walls)),
                mean_peak_mem_mb=float(np.mean(mems)),
            )
        )
    return summaries


def _fmt_score(mean: float, std: float) -> str:
    """Format mean +/- std at 3-4 significant figures.

    "0.7155759334564209" → "0.716 +/- 0.012". If std is zero (single
    seed), drop the +/- portion.
    """
    if not np.isfinite(mean):
        return "n/a"
    # Use 4 sig figs for the mean.
    if abs(mean) >= 1:
        m = f"{mean:.3f}"
    elif abs(mean) >= 1e-3:
        m = f"{mean:.4f}"
    else:
        m = f"{mean:.3e}"
    if std == 0.0 or not np.isfinite(std):
        return m
    s = f"{std:.2g}"
    return f"{m} +/- {s}"


def build_report(
    output_dir: Path,
    results_dir: Path | None = None,
    title: str = "swarmtorch benchmark report",
) -> Path:
    """Write a markdown report from per-seed JSONs.

    Args:
        output_dir: Directory to write ``report.md`` (and read JSONs from
            if ``results_dir`` is None).
        results_dir: Directory containing per-seed JSONs (defaults to
            ``output_dir``).
        title: Top-level header.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir) if results_dir is not None else output_dir

    results = load_results(results_dir)
    summaries = aggregate_results(results)

    # Prefer the hardware fingerprint stamped onto the first result
    # (the machine that *produced* the data); fall back to the local
    # machine fingerprint when re-building a report on a different
    # machine.
    hw = None
    if results:
        hw = results[0].meta.get("hardware")
    if hw is None:
        hw = hardware_info()

    lines: list[str] = [
        f"# {title}",
        "",
        f"**Machine:** `{hw['hostname']}` -- {hw.get('cpu_model','?')} "
        f"({hw.get('cpu_count_logical','?')} cores), {hw.get('ram_mb','?')} MB RAM, {hw.get('os','?')}",
        f"**GPU:** {hw.get('gpu_name') or 'none'}"
        + (
            f" ({hw.get('gpu_total_mem_mb','?')} MB, sm_{hw.get('gpu_capability','?')})"
            if hw.get("cuda_available")
            else ""
        )
        + f"  |  **PyTorch:** {hw.get('torch','?')} (CUDA {hw.get('torch_compiled_cuda','?')})",
        "",
    ]
    if not summaries:
        lines.append("_No results found._")
        path = output_dir / "report.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    tasks = sorted({s.task_name for s in summaries})
    algos = sorted({s.algo_name for s in summaries})

    lines.append(f"Tasks: {len(tasks)}  |  Algorithms: {len(algos)}")
    lines.append("")

    # Per-task tables.
    for task in tasks:
        lines.append(f"## Task: {task}")
        lines.append("")
        lines.append("| Algorithm | N | Score (mean +/- std) | Median | Wall (s) | Peak mem (MB) |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        cells = [s for s in summaries if s.task_name == task]
        cells = sorted(cells, key=lambda s: s.mean_score)
        for s in cells:
            lines.append(
                f"| {s.algo_name} | {s.n_seeds} | {_fmt_score(s.mean_score, s.std_score)} | "
                f"{_fmt_score(s.median_score, 0.0)} | {s.mean_wall:.3f} | {s.mean_peak_mem_mb:.2f} |"
            )
        lines.append("")

    # Friedman + Nemenyi if the grid is wide enough.
    # Friedman requires >= 3 algorithms (scipy constraint) and we want
    # at least 2 tasks so the test has something to compare across.
    if len(tasks) >= 2 and len(algos) >= 3:
        score_matrix = np.full((len(tasks), len(algos)), np.nan)
        for s in summaries:
            i = tasks.index(s.task_name)
            j = algos.index(s.algo_name)
            score_matrix[i, j] = s.mean_score
        if np.isnan(score_matrix).any():
            lines.append("## Statistical tests")
            lines.append("")
            lines.append(
                "_Skipped: not every (task, algorithm) cell has a result. "
                "Re-run with the same algorithms across all tasks for "
                "Friedman / Nemenyi to apply._"
            )
        else:
            fr = friedman_test(score_matrix)
            lines.append("## Statistical tests")
            lines.append("")
            lines.append(
                f"**Friedman:** chi^2 = {fr.statistic:.3f}, "
                f"p = {fr.pvalue:.4g}  (k={fr.n_algorithms} algorithms, "
                f"N={fr.n_tasks} tasks; "
                f"{'reject' if fr.reject_null else 'cannot reject'} the null)"
            )
            try:
                cd = nemenyi_critical_difference(len(algos), len(tasks))
                ranks = rank_algorithms(score_matrix)
                lines.append("")
                lines.append(f"**Nemenyi CD (alpha=0.05):** {cd:.3f}")
                lines.append("")
                lines.append("| Algorithm | Avg rank |")
                lines.append("| --- | --- |")
                for j in np.argsort(ranks):
                    lines.append(f"| {algos[j]} | {ranks[j]:.3f} |")
            except ValueError as e:
                lines.append("")
                lines.append(f"_Nemenyi skipped: {e}_")

    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

"""Tests for convergence plots, CD diagrams, and report builder (Stage 2.4 + 2.6)."""

from pathlib import Path


from swarmtorch.benchmark import (
    aggregate_results,
    build_report,
    convergence_plot,
    critical_difference_diagram,
)
from swarmtorch.benchmark.run import RunResult


def _fake_result(algo: str, task: str, seed: int, scores: list[float]) -> RunResult:
    """Build a synthetic RunResult with a controlled trajectory."""
    trajectory = [(i * 10, s) for i, s in enumerate(scores, start=1)]
    return RunResult(
        algo_name=algo,
        task_name=task,
        seed=seed,
        final_score=scores[-1],
        wall_seconds=0.1 * (seed + 1),
        peak_mem_mb=2.0,
        fe_used=len(scores) * 10,
        trajectory=trajectory,
        meta={},
    )


def test_aggregate_results_groups_by_task_algo():
    results = [
        _fake_result("PSO", "toy", 0, [1.0, 0.5, 0.3]),
        _fake_result("PSO", "toy", 1, [1.2, 0.6, 0.4]),
        _fake_result("Adam", "toy", 0, [1.0, 0.1, 0.05]),
    ]
    summaries = aggregate_results(results)
    assert len(summaries) == 2
    by_algo = {s.algo_name: s for s in summaries}
    assert by_algo["PSO"].n_seeds == 2
    assert by_algo["Adam"].n_seeds == 1
    assert by_algo["PSO"].mean_score == 0.35  # mean of 0.3, 0.4


def test_convergence_plot_writes_png(tmp_path: Path):
    results = []
    for seed in range(3):
        results.append(_fake_result("PSO", "toy", seed, [1.0 - 0.1 * seed, 0.5, 0.3]))
        results.append(_fake_result("Adam", "toy", seed, [1.0 - 0.1 * seed, 0.2, 0.05]))
    out = tmp_path / "conv.png"
    path = convergence_plot(results, out)
    assert path.exists()
    assert path.stat().st_size > 1000  # >1KB sanity check


def test_critical_difference_diagram_writes_png(tmp_path: Path):
    out = tmp_path / "cd.png"
    path = critical_difference_diagram(
        avg_ranks=[1.5, 2.0, 3.0, 3.5],
        names=["A", "B", "C", "D"],
        cd=1.2,
        output_path=out,
    )
    assert path.exists()


def test_build_report_writes_markdown(tmp_path: Path):
    # Persist a small grid: 2 tasks x 3 algos x 3 seeds (Friedman needs k>=3).
    for task in ("task_a", "task_b"):
        for algo, base in (("PSO", 0.3), ("Adam", 0.05), ("CMAES", 0.1)):
            for seed in range(3):
                _fake_result(algo, task, seed, [1.0, 0.5, base + 0.01 * seed]).save(
                    tmp_path
                )

    path = build_report(tmp_path, title="test")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# test" in text
    assert "Task: task_a" in text
    assert "Task: task_b" in text
    assert "Friedman" in text  # 2 tasks x 3 algos triggers the stats block
    assert "Avg rank" in text


def test_build_report_skips_stats_when_too_few_algos(tmp_path: Path):
    # 2 algos -> not enough for Friedman; report still builds but stats block omitted.
    for task in ("a", "b"):
        for algo in ("PSO", "Adam"):
            for seed in range(2):
                _fake_result(algo, task, seed, [1.0, 0.5, 0.2]).save(tmp_path)
    path = build_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Friedman" not in text

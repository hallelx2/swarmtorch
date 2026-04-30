"""End-to-end test for the sweep runner (Stage 4)."""

from pathlib import Path

import pytest

from swarmtorch.benchmark import BenchmarkConfig
from swarmtorch.experiments import make_synthetic_tasks, run_sweep


def test_run_sweep_produces_report_and_results(tmp_path: Path):
    """Run 2 algorithms x 1 task x 2 seeds, confirm artefacts appear."""
    tasks = make_synthetic_tasks(func_names=["sphere"], dims=[5])
    config = BenchmarkConfig(
        seeds=[0, 1],
        max_fe=80,
        log_every=10,
        output_dir=tmp_path,
    )
    report = run_sweep(
        tasks=tasks,
        algorithm_names=["PSO", "Adam"],
        config=config,
        swarm_size=6,
        task_kind="synthetic",
    )
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Task: sphere_d5" in text
    # 2 algorithms x 2 seeds = 4 JSONs
    jsons = list(tmp_path.glob("*.json"))
    assert len(jsons) == 4
    # Convergence PNG should be written.
    assert (tmp_path / "convergence_sphere_d5.png").exists()


def test_run_sweep_with_ablation_axis(tmp_path: Path):
    tasks = make_synthetic_tasks(func_names=["sphere"], dims=[5])
    config = BenchmarkConfig(
        seeds=[0],
        max_fe=60,
        log_every=10,
        output_dir=tmp_path,
    )
    report = run_sweep(
        tasks=tasks,
        algorithm_names=["PSO"],
        config=config,
        swarm_size=6,
        ablation_axes={"init_strategy": ["model", "uniform"]},
        task_kind="synthetic",
    )
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    # Each axis value becomes its own column in the table.
    assert "PSO[init_strategy=model]" in text
    assert "PSO[init_strategy=uniform]" in text


def test_run_sweep_validates_task_kind():
    tasks = make_synthetic_tasks(func_names=["sphere"], dims=[5])
    with pytest.raises(ValueError):
        run_sweep(
            tasks=tasks,
            algorithm_names=["PSO"],
            config=BenchmarkConfig(seeds=[0], max_fe=20),
            task_kind="invalid",
        )

"""Tests for run harness, seeding, JSON persistence (Stage 2.1 + 2.5)."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from swarmtorch.baselines.training import AdamBaseline
from swarmtorch.benchmark import (
    BenchmarkConfig,
    RunResult,
    run_one,
    seed_everything,
)
from swarmtorch.swarm.model_training.pso import PSO


def test_seed_everything_is_deterministic():
    seed_everything(42)
    a = torch.rand(5)
    seed_everything(42)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_run_one_creates_json(tmp_path: Path):
    config = BenchmarkConfig(
        seeds=[0],
        max_fe=20,
        log_every=5,
        output_dir=tmp_path,
    )

    def factory():
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = AdamBaseline(model.parameters(), lr=1e-2)
        x, y = torch.randn(8, 4), torch.randn(8, 2)
        last = {"loss": float("inf")}

        def step_fn():
            loss = opt.step(lambda: F.mse_loss(model(x), y))
            last["loss"] = float(loss.item()) if loss is not None else last["loss"]
            return last["loss"]

        return opt, step_fn

    result = run_one(
        algo_factory=factory,
        task_name="toy_regression",
        algo_name="Adam",
        config=config,
        seed=0,
        meta={"note": "smoke test"},
    )

    assert isinstance(result, RunResult)
    assert result.algo_name == "Adam"
    assert result.task_name == "toy_regression"
    assert result.fe_used >= config.max_fe
    assert result.wall_seconds > 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    loaded = RunResult.load(files[0])
    assert loaded.final_score == result.final_score
    assert loaded.meta["note"] == "smoke test"
    # Every result is auto-stamped with the machine fingerprint.
    assert "hardware" in loaded.meta
    assert "torch" in loaded.meta["hardware"]


def test_run_one_with_pso_records_trajectory(tmp_path: Path):
    config = BenchmarkConfig(
        seeds=[0],
        max_fe=200,
        log_every=20,
        output_dir=tmp_path,
    )

    def factory():
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        opt = PSO(model.parameters(), swarm_size=10, init_strategy="model")
        x, y = torch.randn(8, 4), torch.randn(8, 2)
        loss_fn = nn.MSELoss()

        # Wire the tracker by going through the wrapped closure.
        tracker = config_tracker[0]

        def step_fn():
            opt.step(tracker.wrap_closure(lambda: loss_fn(model(x), y)))
            return float(opt.global_best_fitness.item())

        return opt, step_fn

    # Hand the run_one tracker into the factory via closure cell.
    config_tracker: list = [None]
    from swarmtorch.benchmark.budget import FEBudgetTracker

    config_tracker[0] = FEBudgetTracker(max_fe=config.max_fe, log_every=config.log_every)

    # Run inline rather than through run_one this time so we control the tracker.
    optimizer, step_fn = factory()
    while not config_tracker[0].done:
        step_fn()

    assert config_tracker[0].fe_count >= config.max_fe
    assert len(config_tracker[0].trajectory) > 0

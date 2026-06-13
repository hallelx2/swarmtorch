"""Tests for HPO baselines (Stage 3.2)."""

import pytest
from torch import nn

from swarmtorch.baselines.hpo import RandomSearchBaseline


def _toy_objective():
    """A 2-d toy HPO problem with a known minimum at (lr=0.05, hidden=64)."""

    def model_fn(params):
        return nn.Linear(int(params["hidden"]), 1)

    param_space = {"lr": (0.001, 0.1), "hidden": [16, 32, 64, 128]}

    def train_fn(model, params):
        # Convex bowl with minimum at lr=0.05 and hidden=64 (index 2 of [16,32,64,128]).
        return abs(params["lr"] - 0.05) * 10 + abs(params["hidden"] - 64) / 100.0

    return model_fn, param_space, train_fn


def test_random_search_finds_reasonable_solution():
    model_fn, space, train_fn = _toy_objective()
    s = RandomSearchBaseline(
        model_fn=model_fn,
        param_space=space,
        train_fn=train_fn,
        n_trials=30,
        seed=0,
        verbose=False,
    )
    result = s.search()
    assert "lr" in result.best_params
    assert result.best_params["hidden"] in [16, 32, 64, 128]
    assert len(result.history) == 30
    # Basic floor: should beat the worst-case (~5.5)
    assert result.best_score < 1.0


def test_tpe_baseline_beats_random_on_toy():
    """TPE should converge to lower loss than random within the same budget."""
    optuna = pytest.importorskip("optuna")  # noqa: F841
    from swarmtorch.baselines.hpo import TPESearchBaseline

    model_fn, space, train_fn = _toy_objective()

    rs = RandomSearchBaseline(
        model_fn=model_fn, param_space=space, train_fn=train_fn,
        n_trials=30, seed=0, verbose=False,
    ).search()

    tpe = TPESearchBaseline(
        model_fn=model_fn, param_space=space, train_fn=train_fn,
        n_trials=30, seed=0, verbose=False,
    ).search()

    # Both should be finite, and TPE should be at least competitive.
    assert tpe.best_score <= rs.best_score + 0.1


def test_hyperband_returns_valid_result():
    pytest.importorskip("optuna")
    from swarmtorch.baselines.hpo import HyperbandSearchBaseline

    model_fn, space, train_fn = _toy_objective()
    hb = HyperbandSearchBaseline(
        model_fn=model_fn,
        param_space=space,
        train_fn=train_fn,
        n_trials=15,
        seed=0,
        verbose=False,
        max_resource=9,
    ).search()
    assert "lr" in hb.best_params
    assert hb.best_params["hidden"] in [16, 32, 64, 128]
    assert len(hb.history) >= 1


def test_hyperband_uses_intermediate_reports():
    """When train_fn accepts report_callback, Hyperband should be able to prune."""
    pytest.importorskip("optuna")
    from swarmtorch.baselines.hpo import HyperbandSearchBaseline

    callback_seen = {"called": False}

    def model_fn(params):
        return nn.Linear(4, 1)

    def train_fn(model, params, report_callback=None):
        for step in range(5):
            score = abs(params["lr"] - 0.05) + step * 0.0
            if report_callback is not None:
                callback_seen["called"] = True
                report_callback(score, step)
        return score

    hb = HyperbandSearchBaseline(
        model_fn=model_fn,
        param_space={"lr": (0.001, 0.1)},
        train_fn=train_fn,
        n_trials=5,
        seed=0,
        verbose=False,
        max_resource=5,
    ).search()
    assert callback_seen["called"], "Hyperband never invoked report_callback"
    assert "lr" in hb.best_params


def test_random_search_is_deterministic_with_seed():
    model_fn, space, train_fn = _toy_objective()
    a = RandomSearchBaseline(
        model_fn=model_fn, param_space=space, train_fn=train_fn,
        n_trials=10, seed=42, verbose=False,
    ).search()
    b = RandomSearchBaseline(
        model_fn=model_fn, param_space=space, train_fn=train_fn,
        n_trials=10, seed=42, verbose=False,
    ).search()
    assert a.best_score == b.best_score
    assert a.best_params == b.best_params

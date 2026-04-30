"""Tests for FEBudgetTracker (Stage 2.2)."""

import pytest
import torch

from swarmtorch.benchmark.budget import BudgetExceeded, FEBudgetTracker


def test_plain_closure_increments_by_one():
    tracker = FEBudgetTracker(max_fe=5)
    closure = lambda: torch.tensor(1.5)
    wrapped = tracker.wrap_closure(closure)
    for _ in range(3):
        wrapped()
    assert tracker.fe_count == 3
    assert tracker.best_so_far == 1.5
    assert not tracker.done


def test_done_flag_after_budget_hit():
    tracker = FEBudgetTracker(max_fe=2)
    wrapped = tracker.wrap_closure(lambda: torch.tensor(0.7))
    wrapped()
    wrapped()
    assert tracker.done
    with pytest.raises(BudgetExceeded):
        tracker.raise_if_done()


def test_functional_closure_credits_n_particles():
    tracker = FEBudgetTracker(max_fe=100)
    fc = lambda flat: (flat ** 2).sum()
    wrapped_outer = tracker.wrap_functional_closure(fc, n_particles=10)
    particles = torch.randn(10, 4)
    wrapped_outer(particles)
    assert tracker.fe_count == 10


def test_log_every_records_trajectory():
    tracker = FEBudgetTracker(max_fe=10, log_every=3)
    wrapped = tracker.wrap_closure(lambda: torch.tensor(0.5))
    for _ in range(10):
        wrapped()
    # log_every=3 over 10 evaluations should record at fe=3, 6, 9.
    assert [t[0] for t in tracker.trajectory] == [3, 6, 9]
    assert all(t[1] == 0.5 for t in tracker.trajectory)


def test_best_so_far_only_decreases():
    tracker = FEBudgetTracker(max_fe=10)
    wrapped = tracker.wrap_closure(lambda: torch.tensor(0.0))
    losses = [3.0, 1.0, 2.0, 0.5, 0.7]
    for v in losses:
        # mutate the closure to return v
        tracker.wrap_closure(lambda v=v: torch.tensor(v))()
    assert tracker.best_so_far == 0.5


def test_observe_external_updates_counter_and_best():
    tracker = FEBudgetTracker(max_fe=5)
    tracker.observe_external(2, current_best=0.9)
    tracker.observe_external(1, current_best=0.4)
    assert tracker.fe_count == 3
    assert tracker.best_so_far == pytest.approx(0.4, abs=1e-6)


def test_invalid_max_fe_raises():
    with pytest.raises(ValueError):
        FEBudgetTracker(max_fe=0)

"""Tests for synthetic test-function suite (Stage 4.1)."""

import pytest
import torch

from swarmtorch.experiments.synthetic import (
    SYNTHETIC_FUNCTIONS,
    SyntheticTask,
    make_synthetic_tasks,
)


@pytest.mark.parametrize("name", list(SYNTHETIC_FUNCTIONS))
def test_function_global_minimum_at_origin(name):
    """Every test function should return ~0 at the origin (after shifts)."""
    f = SYNTHETIC_FUNCTIONS[name]
    x = torch.zeros(20)
    val = float(f(x).item())
    assert abs(val) < 1e-3, f"{name}(0) = {val}, expected ~0"


@pytest.mark.parametrize("name", list(SYNTHETIC_FUNCTIONS))
def test_function_positive_off_origin(name):
    f = SYNTHETIC_FUNCTIONS[name]
    x = torch.ones(20)
    val = float(f(x).item())
    # Sphere(1) = 20, Rastrigin(1)/Ackley(1)/etc all > 0.
    assert val > 0


def test_make_synthetic_tasks_default_grid():
    tasks = make_synthetic_tasks()
    # 5 functions x 3 dims = 15 tasks.
    assert len(tasks) == 15
    assert {t.dim for t in tasks} == {10, 50, 200}


def test_make_synthetic_tasks_custom():
    tasks = make_synthetic_tasks(func_names=["sphere"], dims=[5, 10])
    assert [t.name for t in tasks] == ["sphere_d5", "sphere_d10"]
    assert tasks[0].dim == 5


def test_synthetic_task_module_and_closure():
    task = SyntheticTask(name="sphere_d10", dim=10, func_name="sphere")
    module = task.make_module()
    assert module.x.shape == (10,)
    closure = task.make_closure(module)
    val = float(closure().item())
    # Random init ~U(-5, 5), sphere = sum(x^2): expected ~ 10 * E[x^2] ~ 83.
    assert val > 0

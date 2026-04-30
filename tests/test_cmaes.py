"""Tests for the CMA-ES optimizer (Stage 1.8)."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

cma = pytest.importorskip("cma")

from swarmtorch.evolutionary.model_training.cmaes import CMAES  # noqa: E402


def test_cmaes_minimizes_quadratic():
    """CMA-ES should drive a tiny quadratic loss towards zero in a few steps."""
    torch.manual_seed(0)
    target = torch.tensor([1.0, -2.0, 0.5, 3.0])
    w = nn.Parameter(torch.zeros_like(target))

    opt = CMAES([w], swarm_size=12, sigma0=1.0)

    def closure():
        return ((w - target) ** 2).sum()

    initial = closure().item()
    for _ in range(30):
        opt.step(closure)

    assert opt.best_fitness.item() < initial * 0.01, (
        f"CMA-ES failed to reduce quadratic loss meaningfully: "
        f"initial={initial:.4f}, best={opt.best_fitness.item():.4f}"
    )


def test_cmaes_pulls_default_popsize_when_unspecified():
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    opt = CMAES(model.parameters())  # swarm_size left to pycma's default
    x, y = torch.randn(4, 8), torch.randn(4, 4)

    opt.step(lambda: F.mse_loss(model(x), y))

    # pycma's default for d ~ 36 is 4 + floor(3 * ln(36)) = 14.
    assert opt.swarm_size > 0
    assert opt.positions.shape == (opt.swarm_size, sum(p.numel() for p in model.parameters()))


def test_cmaes_with_functional_closure():
    """vmap fast path should work the same way as for swarm-based optimizers."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    x, y = torch.randn(16, 8), torch.randn(16, 4)

    opt = CMAES(model.parameters(), swarm_size=10)
    opt.set_functional_closure(model, lambda fwd: F.mse_loss(fwd(x), y))

    initial = F.mse_loss(model(x), y).item()
    for _ in range(15):
        opt.step()
    final = F.mse_loss(model(x), y).item()

    assert final <= initial + 1e-4
    assert torch.isfinite(opt.positions).all()


def test_cmaes_search_via_generic_search():
    """End-to-end HPO via the CMAESSearch wrapper."""
    from swarmtorch.evolutionary.hyperparameter_tuning import CMAESSearch

    def build_model(p):
        return nn.Linear(p["in_features"], 2)

    param_space = {"in_features": [10, 20, 30], "lr": (0.001, 0.1)}

    def train_fn(model, params):
        return abs(params["lr"] - 0.05) + abs(params["in_features"] - 20)

    search = CMAESSearch(
        model_fn=build_model,
        param_space=param_space,
        train_fn=train_fn,
        iterations=5,
        swarm_size=8,
        device="cpu",
        verbose=False,
    )
    best = search.search()
    assert "lr" in best
    assert "in_features" in best

"""Tests for vmap-vectorized fitness evaluation (Stage 1.2)."""

import time

import torch
import torch.nn.functional as F
from torch import nn

from swarmtorch.swarm.model_training.pso import PSO


def _make_setup(in_features=8, out_features=4, batch=32, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(in_features, 16),
        nn.ReLU(),
        nn.Linear(16, out_features),
    )
    x = torch.randn(batch, in_features)
    y = torch.randn(batch, out_features)
    return model, x, y


def test_functional_closure_matches_loop_path():
    model, x, y = _make_setup()

    opt = PSO(model.parameters(), swarm_size=6, init_strategy="model")
    opt._init_swarm()
    particles = opt.positions.clone()

    # Loop path
    plain_closure = lambda: F.mse_loss(model(x), y)
    fitness_loop = opt._evaluate_fitness(particles, plain_closure)

    # Functional vmap path
    opt.set_functional_closure(model, lambda forward: F.mse_loss(forward(x), y))
    fitness_vmap = opt._evaluate_fitness(particles)

    assert fitness_loop.shape == fitness_vmap.shape
    assert torch.allclose(fitness_loop, fitness_vmap, atol=1e-5, rtol=1e-4), (
        f"loop={fitness_loop}\nvmap={fitness_vmap}"
    )


def test_functional_closure_can_be_disabled():
    model, x, y = _make_setup()
    opt = PSO(model.parameters(), swarm_size=4)
    opt._init_swarm()

    opt.set_functional_closure(model, lambda fwd: F.mse_loss(fwd(x), y))
    assert opt._functional_closure is not None

    opt.set_functional_closure(None, None)
    assert opt._functional_closure is None


def test_pso_step_with_functional_closure_trains():
    """End-to-end: PSO with vmap fitness should reduce loss over a few steps."""
    model, x, y = _make_setup()
    opt = PSO(model.parameters(), swarm_size=10, init_strategy="model")
    opt.set_functional_closure(model, lambda fwd: F.mse_loss(fwd(x), y))

    initial_loss = F.mse_loss(model(x), y).item()
    for _ in range(5):
        opt.step()  # closure can be omitted — functional path doesn't need it
    final_loss = F.mse_loss(model(x), y).item()

    assert final_loss <= initial_loss + 1e-4


def test_vmap_path_is_faster_than_loop():
    """Sanity check that the vectorized path actually wins on swarm_size."""
    model, x, y = _make_setup(in_features=32, out_features=16, batch=64)

    opt = PSO(model.parameters(), swarm_size=32, init_strategy="model")
    opt._init_swarm()
    particles = opt.positions.clone()

    plain_closure = lambda: F.mse_loss(model(x), y)
    # Warm up
    opt._evaluate_fitness(particles, plain_closure)

    t0 = time.perf_counter()
    for _ in range(3):
        opt._evaluate_fitness(particles, plain_closure)
    t_loop = time.perf_counter() - t0

    opt.set_functional_closure(model, lambda fwd: F.mse_loss(fwd(x), y))
    # Warm up vmap
    opt._evaluate_fitness(particles)

    t0 = time.perf_counter()
    for _ in range(3):
        opt._evaluate_fitness(particles)
    t_vmap = time.perf_counter() - t0

    # Allow generous slack — even on CPU vmap should not be dramatically
    # slower than the loop, and on most setups it will be faster.
    assert t_vmap < t_loop * 2.0, (
        f"vmap path unexpectedly slow: loop={t_loop:.3f}s vmap={t_vmap:.3f}s"
    )

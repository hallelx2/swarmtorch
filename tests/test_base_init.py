"""Tests for SwarmOptimizer init strategies (Stage 1.1)."""

import pytest
import torch
from torch import nn

from swarmtorch.swarm.model_training.pso import PSO


@pytest.fixture
def simple_model():
    torch.manual_seed(0)
    return nn.Linear(8, 4)


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.flatten() for p in model.parameters()])


def test_model_init_seeds_first_particle_to_model_weights(simple_model):
    init_weights = _flat_params(simple_model).clone()
    opt = PSO(simple_model.parameters(), swarm_size=5, init_strategy="model", init_sigma=0.1)
    # Trigger _init_swarm without running a full step.
    opt._init_swarm()
    assert opt.positions.shape == (5, init_weights.numel())
    # positions[0] must be exactly the model's current (Kaiming-initialized) weights.
    assert torch.allclose(opt.positions[0], init_weights)


def test_model_init_other_particles_are_perturbed(simple_model):
    opt = PSO(simple_model.parameters(), swarm_size=10, init_strategy="model", init_sigma=0.1)
    opt._init_swarm()
    # Particles 1..n must differ from particle 0.
    diffs = (opt.positions[1:] - opt.positions[0]).abs().sum(dim=1)
    assert (diffs > 0).all()


def test_uniform_init_reproduces_legacy(simple_model):
    opt = PSO(simple_model.parameters(), swarm_size=5, init_strategy="uniform")
    opt._init_swarm()
    assert opt.positions.min() >= 0.0
    assert opt.positions.max() <= 1.0


def test_gaussian_init_zero_mean(simple_model):
    torch.manual_seed(42)
    opt = PSO(simple_model.parameters(), swarm_size=2000, init_strategy="gaussian", init_sigma=0.5)
    opt._init_swarm()
    assert abs(opt.positions.mean().item()) < 0.05  # ~0 mean
    assert abs(opt.positions.std().item() - 0.5) < 0.05


def test_invalid_init_strategy_raises(simple_model):
    with pytest.raises(ValueError):
        PSO(simple_model.parameters(), init_strategy="bogus")


def test_pso_step_does_not_explode_with_model_init(simple_model):
    """Regression: PSO should produce finite weights after a few steps."""
    opt = PSO(simple_model.parameters(), swarm_size=8, init_strategy="model")
    x = torch.randn(16, 8)
    y = torch.randn(16, 4)
    loss_fn = nn.MSELoss()

    def closure():
        return loss_fn(simple_model(x), y)

    for _ in range(3):
        opt.step(closure)

    final = _flat_params(simple_model)
    assert torch.isfinite(final).all()

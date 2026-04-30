"""Tests for swarm-state save/load round-trip (Stage 1.5)."""

import torch
from torch import nn

from swarmtorch.swarm.model_training.pso import PSO


def _model():
    torch.manual_seed(0)
    return nn.Linear(6, 3)


def test_state_dict_contains_swarm_state():
    model = _model()
    opt = PSO(model.parameters(), swarm_size=5)
    x, y = torch.randn(4, 6), torch.randn(4, 3)
    loss_fn = nn.MSELoss()
    opt.step(lambda: loss_fn(model(x), y))

    sd = opt.state_dict()
    assert "_swarm_state" in sd
    swarm = sd["_swarm_state"]
    assert "positions" in swarm
    assert "velocities" in swarm
    assert "personal_best_positions" in swarm
    assert "global_best_position" in swarm


def test_state_dict_round_trip_preserves_positions():
    model = _model()
    opt = PSO(model.parameters(), swarm_size=5)
    x, y = torch.randn(4, 6), torch.randn(4, 3)
    loss_fn = nn.MSELoss()
    for _ in range(3):
        opt.step(lambda: loss_fn(model(x), y))

    sd = opt.state_dict()

    # Fresh optimizer on a fresh model, restored from sd.
    model2 = _model()
    opt2 = PSO(model2.parameters(), swarm_size=5)
    opt2.load_state_dict(sd)

    assert torch.allclose(opt.positions, opt2.positions)
    assert torch.allclose(opt.velocities, opt2.velocities)
    assert torch.allclose(opt.global_best_position, opt2.global_best_position)
    assert torch.allclose(opt.personal_best_fitness, opt2.personal_best_fitness)

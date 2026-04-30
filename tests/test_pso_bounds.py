"""Tests for PSO velocity / position clipping (Stage 1.3)."""

import torch
from torch import nn

from swarmtorch.swarm.model_training.pso import PSO


def test_positions_clipped_to_configured_range():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = PSO(
        model.parameters(),
        swarm_size=10,
        position_clip=2.0,
        velocity_clip=0.5,
        init_strategy="model",
    )
    x, y = torch.randn(8, 4), torch.randn(8, 2)
    loss_fn = nn.MSELoss()
    for _ in range(20):
        opt.step(lambda: loss_fn(model(x), y))

    bound = 2.0 * opt._init_scale
    assert opt.positions.abs().max().item() <= bound + 1e-6


def test_velocities_clipped():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = PSO(
        model.parameters(),
        swarm_size=10,
        position_clip=10.0,
        velocity_clip=0.3,
        init_strategy="model",
    )
    x, y = torch.randn(8, 4), torch.randn(8, 2)
    loss_fn = nn.MSELoss()
    for _ in range(10):
        opt.step(lambda: loss_fn(model(x), y))

    bound = 0.3 * opt._init_scale
    assert opt.velocities.abs().max().item() <= bound + 1e-6


def test_clipping_disabled_by_none():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    opt = PSO(
        model.parameters(),
        swarm_size=5,
        position_clip=None,
        velocity_clip=None,
    )
    x, y = torch.randn(8, 4), torch.randn(8, 2)
    loss_fn = nn.MSELoss()
    # Should run without error and not blow up to infinity in 5 steps.
    for _ in range(5):
        opt.step(lambda: loss_fn(model(x), y))
    assert torch.isfinite(opt.positions).all()

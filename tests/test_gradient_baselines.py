"""Tests for the gradient training baselines (Stage 3.1)."""

import torch
import torch.nn.functional as F
from torch import nn

from swarmtorch.baselines.training import (
    AdamBaseline,
    AdamWBaseline,
    LBFGSBaseline,
    RMSpropBaseline,
    SGDBaseline,
)


def _make_setup():
    torch.manual_seed(0)
    model = nn.Linear(8, 4)
    x = torch.randn(64, 8)
    # Linearly-generated target so MSE has a true zero.
    true_w = torch.randn(4, 8)
    true_b = torch.randn(4)
    y = x @ true_w.T + true_b
    return model, x, y


def test_adam_baseline_drives_loss_down():
    model, x, y = _make_setup()
    opt = AdamBaseline(model.parameters(), lr=0.05)

    initial = F.mse_loss(model(x), y).item()
    for _ in range(500):
        opt.step(lambda: F.mse_loss(model(x), y))
    final = F.mse_loss(model(x), y).item()

    # Linear regression on a linear target with 500 Adam steps should drive
    # loss roughly to zero — use a generous 1% threshold.
    assert final < initial * 0.01, f"Adam failed to fit: initial={initial:.3f} final={final:.4f}"


def test_lbfgs_baseline_uses_closure_path():
    """LBFGS re-evaluates the closure internally; wrapper must forward it."""
    model, x, y = _make_setup()
    opt = LBFGSBaseline(model.parameters(), lr=1.0, max_iter=20)

    initial = F.mse_loss(model(x), y).item()
    for _ in range(5):
        opt.step(lambda: F.mse_loss(model(x), y))
    final = F.mse_loss(model(x), y).item()

    assert final < initial


def test_set_functional_closure_is_noop():
    model, _, _ = _make_setup()
    opt = AdamBaseline(model.parameters())
    # Should not raise even though gradient baselines don't use vmap.
    opt.set_functional_closure(model, lambda fwd: torch.tensor(0.0))


def test_all_gradient_classes_construct():
    model = nn.Linear(4, 2)
    AdamBaseline(model.parameters())
    AdamWBaseline(model.parameters())
    SGDBaseline(model.parameters())
    RMSpropBaseline(model.parameters())
    LBFGSBaseline(model.parameters())


def test_state_dict_round_trip():
    model = nn.Linear(4, 2)
    opt = AdamBaseline(model.parameters(), lr=1e-3)
    x, y = torch.randn(8, 4), torch.randn(8, 2)
    opt.step(lambda: F.mse_loss(model(x), y))

    sd = opt.state_dict()
    model2 = nn.Linear(4, 2)
    opt2 = AdamBaseline(model2.parameters(), lr=1e-3)
    opt2.load_state_dict(sd)
    # Adam stores running first/second moments; after the load both should match.
    assert opt.state_dict()["state"].keys() == opt2.state_dict()["state"].keys()

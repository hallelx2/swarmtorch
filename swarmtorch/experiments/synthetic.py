"""Classical black-box test functions for the dimensionality-wall study.

These are the standard CEC-style benchmarks every metaheuristic paper
uses. We implement them as differentiable PyTorch functions so the same
task definition can drive both gradient methods and population-based
optimizers — and so the vmap fast path works without modification.

All functions have global minimum value 0 at the origin (Rosenbrock at
``x = 1``, but we shift it during evaluation so it's also at the origin
for consistency in plots).

References:
    Suganthan et al. (2017). Problem Definitions and Evaluation Criteria
        for the CEC 2017 Special Session on Real-Parameter Optimization.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


# --- Test functions: take flat parameter tensor, return scalar loss. -----


def sphere(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sum(x^2). Convex, separable; gradient methods solve trivially."""
    return (x ** 2).sum()


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """Highly multi-modal, separable. Standard difficult test."""
    a = 10.0
    return a * x.numel() + (x ** 2 - a * torch.cos(2 * math.pi * x)).sum()


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """Narrow curved valley — classic non-separable benchmark.

    Shifted so the minimum is at the origin (subtract 1 from each coordinate).
    """
    y = x + 1.0  # actual rosenbrock minimum at y=1
    return (100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2).sum()


def ackley(x: torch.Tensor) -> torch.Tensor:
    """Multi-modal with a near-flat plateau; punishing for naive search."""
    n = x.numel()
    a = 20.0
    b = 0.2
    c = 2 * math.pi
    s1 = (x ** 2).sum() / n
    s2 = torch.cos(c * x).sum() / n
    return -a * torch.exp(-b * torch.sqrt(s1)) - torch.exp(s2) + a + math.e


def griewank(x: torch.Tensor) -> torch.Tensor:
    """Multi-modal, weakly non-separable, fewer local minima as d grows."""
    sq = (x ** 2).sum() / 4000.0
    idx = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    prod = torch.cos(x / torch.sqrt(idx)).prod()
    return sq - prod + 1.0


SYNTHETIC_FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "ackley": ackley,
    "griewank": griewank,
}


# --- Task adapter ---------------------------------------------------------


class _ParamModule(nn.Module):
    """Wrap a flat parameter tensor as an nn.Module so SwarmOptimizer's
    ``params`` plumbing accepts it and the closure can flow through.
    """

    def __init__(self, dim: int, init_range: float = 5.0) -> None:
        super().__init__()
        # Initialize uniformly within the standard search domain so the
        # initial loss is a meaningful "starting point" — Kaiming would
        # not apply to abstract benchmark variables.
        self.x = nn.Parameter(
            torch.empty(dim).uniform_(-init_range, init_range)
        )

    def forward(self) -> torch.Tensor:
        return self.x


@dataclass
class SyntheticTask:
    """Bundle of (name, dimension, function) for the sweep runner.

    Attributes:
        name: Identifier persisted in result JSONs (e.g. ``"rastrigin_d50"``).
        dim: Problem dimension.
        func_name: Key into :data:`SYNTHETIC_FUNCTIONS`.
        init_range: Half-width of the uniform initialization box.
    """

    name: str
    dim: int
    func_name: str
    init_range: float = 5.0

    def function(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return SYNTHETIC_FUNCTIONS[self.func_name]

    def make_module(self) -> _ParamModule:
        return _ParamModule(self.dim, init_range=self.init_range)

    def make_closure(self, module: _ParamModule) -> Callable[[], torch.Tensor]:
        f = self.function()
        return lambda: f(module.x)

    def make_functional_closure(
        self, module: _ParamModule
    ) -> Callable[[Any], torch.Tensor]:
        """For the vmap fast path: returns a closure that takes a callable
        ``forward`` (yielding the current particle's parameter vector) and
        evaluates the test function on that vector.
        """
        f = self.function()
        return lambda forward: f(forward())


def make_synthetic_tasks(
    func_names: list[str] | None = None,
    dims: list[int] | None = None,
) -> list[SyntheticTask]:
    """Cartesian product of functions × dimensions.

    Defaults: all 5 functions × ``[10, 50, 200]`` (skips d=1000 by default
    to keep CI reasonable; the paper experiments include d=1000).
    """
    func_names = func_names or list(SYNTHETIC_FUNCTIONS)
    dims = dims or [10, 50, 200]
    return [
        SyntheticTask(
            name=f"{fname}_d{d}",
            dim=d,
            func_name=fname,
        )
        for fname in func_names
        for d in dims
    ]

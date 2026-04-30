"""Baseline (non-metaheuristic) optimizers and HPO methods.

This sub-package collects gradient-based optimizers and standard HPO
samplers (Random, TPE, Hyperband) wrapped behind the same interfaces used
by ``swarmtorch``'s metaheuristic optimizers and searchers. The benchmark
harness can then iterate over a heterogeneous list of (metaheuristic,
gradient, Bayesian) candidates without special-casing.
"""

from swarmtorch.baselines.training import (
    GradientBaseline,
    AdamBaseline,
    AdamWBaseline,
    SGDBaseline,
    RMSpropBaseline,
    LBFGSBaseline,
)

__all__ = [
    "GradientBaseline",
    "AdamBaseline",
    "AdamWBaseline",
    "SGDBaseline",
    "RMSpropBaseline",
    "LBFGSBaseline",
]

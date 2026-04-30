"""Curated algorithm short-list and operator-structure taxonomy.

The legacy ``algo_registry.py`` lists all 60+ optimizers by metaphor
("swarm", "bio_inspired", "hybrid"). For the paper we want two things:

1. **A short list** — ``PAPER_ALGORITHMS`` — the best-per-family
   metaheuristics from the existing benchmark, plus CMA-ES, plus the
   gradient baselines. About a dozen entries total. Keeps the headline
   table legible.

2. **A non-metaphorical taxonomy** — ``OPERATOR_TAXONOMY`` — groups
   algorithms by their *operator structure* (velocity-based,
   attractor-based, mutation+selection, density-based, distribution-
   based, gradient). This directly engages the Sorensen 2015 / Aranha
   2022 critique that 2026 reviewers will demand we address.

Each algorithm in ``PAPER_ALGORITHMS`` is paired with a factory that
yields an instance ready to drive ``run_one``. The factory abstraction
lets the sweep runner instantiate algorithms uniformly without caring
whether they're swarm, evolutionary, or gradient.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch

from swarmtorch.baselines.training import (
    AdamBaseline,
    AdamWBaseline,
    LBFGSBaseline,
    RMSpropBaseline,
    SGDBaseline,
)
from swarmtorch.evolutionary.model_training import CA, CEM, CMAES
from swarmtorch.bio_inspired.model_training import SineCosine
from swarmtorch.human_based.model_training import TLBO
from swarmtorch.hybrid.model_training import Gorilla
from swarmtorch.physics.model_training import FPA
from swarmtorch.swarm.model_training import PSO


# Twelve algorithms — paper headline grid (5 metaheuristics + CMA-ES + 5 gradient baselines + Random as floor).
PAPER_ALGORITHMS: dict[str, Callable[..., Any]] = {
    # Strong metaheuristics: best per family from the legacy benchmark.
    "PSO": PSO,
    "CA": CA,
    "CEM": CEM,
    "TLBO": TLBO,
    "FPA": FPA,
    "Gorilla": Gorilla,
    "SineCosine": SineCosine,
    # The metaheuristic that should beat the rest — and the most glaring
    # omission from the legacy benchmark.
    "CMAES": CMAES,
    # Gradient baselines.
    "Adam": AdamBaseline,
    "AdamW": AdamWBaseline,
    "SGD": SGDBaseline,
    "RMSprop": RMSpropBaseline,
    "LBFGS": LBFGSBaseline,
}


# Operator-structure taxonomy. Names are deliberately mechanical, not
# metaphorical. References given to the operator pattern, not the
# branded metaphor.
OPERATOR_TAXONOMY: dict[str, list[str]] = {
    # Particles carry velocity vectors and update via attraction to remembered
    # personal/global bests (Eberhart & Kennedy 1995).
    "velocity-based": ["PSO"],
    # Population is guided by a small set of "leader" attractors at every step;
    # no velocity, just weighted moves toward leaders.
    "attractor-based": ["TLBO", "Gorilla"],
    # Generate offspring via mutation/crossover; selection keeps the fittest.
    "mutation-selection": ["CA"],
    # Maintain an explicit distribution (mean + covariance) over candidate
    # solutions; sample from it, update the distribution from the best samples.
    "distribution-based": ["CEM", "CMAES"],
    # Levy/Cauchy random walks over the search space modulated by current best.
    "random-walk": ["FPA", "SineCosine"],
    # Gradient-based optimizers — included as baselines, separately taxonomized
    # because their operator is fundamentally different (uses df/dx).
    "gradient": ["Adam", "AdamW", "SGD", "RMSprop", "LBFGS"],
}


# HPO short-list mirrors the training short-list one-for-one.
PAPER_HPO_SEARCHERS: list[str] = [
    "PSOSearch",
    "CASearch",
    "CEMSearch",
    "TLBOSearch",
    "FPASearch",
    "GorillaSearch",
    "SineCosineSearch",
    "CMAESSearch",
    "RandomSearchBaseline",
    "TPESearchBaseline",
    "HyperbandSearchBaseline",
]


# --- Algorithm factory ---------------------------------------------------


def build_algorithm_factory(
    name: str,
    swarm_size: int = 30,
    init_strategy: str = "model",
    extra_kwargs: dict[str, Any] | None = None,
) -> Callable[[torch.nn.Module], Any]:
    """Return a function that, given a ``model``, builds the algorithm.

    The returned callable is a *constructor* — call it inside the
    ``algo_factory`` you pass to :func:`run_one` so seeding takes effect
    and a fresh optimizer is built per (algorithm, task, seed) cell.

    Args:
        name: Key into :data:`PAPER_ALGORITHMS`.
        swarm_size: Population size for metaheuristics. Ignored by
            gradient baselines.
        init_strategy: Forwarded to ``SwarmOptimizer`` subclasses;
            ignored by gradient baselines.
        extra_kwargs: Forwarded to the underlying constructor.
    """
    if name not in PAPER_ALGORITHMS:
        raise KeyError(
            f"{name!r} not in PAPER_ALGORITHMS. Known: {sorted(PAPER_ALGORITHMS)}"
        )
    cls = PAPER_ALGORITHMS[name]
    extra_kwargs = dict(extra_kwargs or {})

    is_swarm = name in {
        algo
        for group, names in OPERATOR_TAXONOMY.items()
        if group != "gradient"
        for algo in names
    }

    # Older subclasses (e.g. CA, FDA, PFA in cem.py) don't accept
    # ``init_strategy`` or ``init_sigma`` in their __init__ — they only
    # forward ``swarm_size`` and ``device`` to ``SwarmOptimizer.__init__``.
    # Introspect the constructor and only pass what it accepts so we don't
    # have to edit 21 subclass files to participate in the new init API.
    sig = inspect.signature(cls.__init__)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    accepts_init_strategy = "init_strategy" in sig.parameters or accepts_kwargs
    accepts_swarm_size = (
        "swarm_size" in sig.parameters
        or "population_size" in sig.parameters
        or accepts_kwargs
    )

    def factory(model: torch.nn.Module) -> Any:
        if is_swarm:
            kwargs: dict[str, Any] = {}
            if accepts_swarm_size:
                # Some classes (CEM, GA) name it population_size.
                key = (
                    "population_size"
                    if "population_size" in sig.parameters
                    and "swarm_size" not in sig.parameters
                    else "swarm_size"
                )
                kwargs[key] = swarm_size
            if accepts_init_strategy:
                kwargs["init_strategy"] = init_strategy
            kwargs.update(extra_kwargs)
            return cls(model.parameters(), **kwargs)
        return cls(model.parameters(), **extra_kwargs)

    return factory


def operator_group(algorithm_name: str) -> str:
    """Return the operator-structure group for a given algorithm."""
    for group, names in OPERATOR_TAXONOMY.items():
        if algorithm_name in names:
            return group
    return "unknown"

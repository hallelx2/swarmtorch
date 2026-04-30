"""Stage 4 paper-grade experiments: synthetic suite, training tasks,
HPO tasks, sweep runner, and the curated algorithm registry.

This sub-package contains *task definitions and the sweep engine* — not
the actual long-running results. To execute the paper experiments,
invoke the scripts in ``scripts/`` (e.g. ``scripts/run_synthetic.py``).
"""

from swarmtorch.experiments.registry import (
    OPERATOR_TAXONOMY,
    PAPER_ALGORITHMS,
    PAPER_HPO_SEARCHERS,
    build_algorithm_factory,
)
from swarmtorch.experiments.runner import run_sweep
from swarmtorch.experiments.synthetic import (
    SYNTHETIC_FUNCTIONS,
    SyntheticTask,
    make_synthetic_tasks,
)

__all__ = [
    "OPERATOR_TAXONOMY",
    "PAPER_ALGORITHMS",
    "PAPER_HPO_SEARCHERS",
    "build_algorithm_factory",
    "run_sweep",
    "SYNTHETIC_FUNCTIONS",
    "SyntheticTask",
    "make_synthetic_tasks",
]

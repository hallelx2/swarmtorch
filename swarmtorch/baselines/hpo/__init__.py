"""HPO baselines: Random Search, TPE (Optuna), Hyperband (Optuna).

Each class exposes a ``search()`` method that returns a dict
``{"best_params", "best_score", "history"}`` and accepts the same
``param_space`` format as :class:`HyperparameterSearch` —
``(min, max)`` for continuous params, ``[v1, v2, ...]`` for categoricals.

Optuna is a soft dependency. The Random baseline does not require it; TPE
and Hyperband do, and the constructor raises a clear error if it's not
installed.
"""

from swarmtorch.baselines.hpo.base import BaselineHPO, HPOResult
from swarmtorch.baselines.hpo.random_search import RandomSearchBaseline
from swarmtorch.baselines.hpo.optuna_search import TPESearchBaseline, HyperbandSearchBaseline

__all__ = [
    "BaselineHPO",
    "HPOResult",
    "RandomSearchBaseline",
    "TPESearchBaseline",
    "HyperbandSearchBaseline",
]

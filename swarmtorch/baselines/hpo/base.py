"""Common interface for HPO baselines."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class HPOResult:
    """Result of an HPO run.

    ``history`` is a list of ``(trial_index, score, params)`` tuples in
    the order trials were evaluated — the raw material for convergence
    curves and statistical tests.
    """

    best_params: dict
    best_score: float
    history: list[tuple[int, float, dict]] = field(default_factory=list)


class BaselineHPO:
    """Abstract base for non-metaheuristic HPO methods.

    The constructor and ``param_space`` format mirror
    :class:`swarmtorch.base.hyperparam_search.HyperparameterSearch` so
    users can swap searchers without changing their search-space dict.

    Subclasses implement :meth:`search` and return an :class:`HPOResult`.
    """

    def __init__(
        self,
        model_fn: Callable[[dict], torch.nn.Module],
        param_space: dict[str, tuple[Any, ...] | list[Any]],
        train_fn: Callable[[torch.nn.Module, dict], float],
        n_trials: int = 50,
        device: str = "cpu",
        verbose: bool = True,
        seed: int | None = None,
    ) -> None:
        self.model_fn = model_fn
        self.param_space = param_space
        self.train_fn = train_fn
        self.n_trials = n_trials
        self.device = torch.device(device)
        self.verbose = verbose
        self.seed = seed

    def _evaluate(self, params: dict) -> float:
        model = self.model_fn(params)
        # Only torch modules need a device move; sklearn / xgboost estimators
        # (and any other non-torch model) are used as-is.
        if hasattr(model, "to"):
            model = model.to(self.device)
        return float(self.train_fn(model, params))

    def search(self) -> HPOResult:
        raise NotImplementedError("Subclasses must implement search()")

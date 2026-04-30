"""TPE and Hyperband HPO baselines built on Optuna.

These are the two "real" baselines paper reviewers expect:

* **TPE** (Tree-structured Parzen Estimator, Bergstra et al. 2011) — the
  Bayesian sampler used by every modern HPO library.
* **Hyperband** (Li et al. 2017) — multi-fidelity bandit that early-stops
  unpromising trials. Requires a ``train_fn`` that yields intermediate
  scores; falls back to single-fidelity behavior otherwise.
"""

from collections.abc import Callable
from typing import Any

import torch

from swarmtorch.baselines.hpo.base import BaselineHPO, HPOResult


def _require_optuna():
    try:
        import optuna  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "TPE and Hyperband baselines require Optuna. Install with: "
            "pip install 'optuna>=3.0'"
        ) from e


def _suggest_params(trial: Any, param_space: dict) -> dict:
    """Translate a swarmtorch param_space into Optuna ``trial.suggest_*`` calls."""
    params: dict[str, Any] = {}
    for key, value in param_space.items():
        if isinstance(value, list):
            params[key] = trial.suggest_categorical(key, value)
        else:
            lo, hi = float(value[0]), float(value[1])
            # Use log scale when the range spans more than two orders of magnitude
            # AND both bounds are positive — typical for learning rates.
            log = lo > 0 and hi / lo >= 100
            params[key] = trial.suggest_float(key, lo, hi, log=log)
    return params


class TPESearchBaseline(BaselineHPO):
    """Bayesian HPO via Optuna's TPE sampler."""

    def search(self) -> HPOResult:
        _require_optuna()
        import optuna

        history: list[tuple[int, float, dict]] = []

        def objective(trial: "optuna.Trial") -> float:
            params = _suggest_params(trial, self.param_space)
            score = self._evaluate(params)
            history.append((trial.number, score, params))
            return score

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        best = study.best_trial
        return HPOResult(
            best_params=dict(best.params),
            best_score=float(best.value),
            history=history,
        )


class HyperbandSearchBaseline(BaselineHPO):
    """Multi-fidelity HPO via Optuna's Hyperband pruner.

    Args:
        train_fn: As in the base class. To benefit from early stopping, the
            function should accept an optional ``report_callback`` keyword
            and call it as ``report_callback(intermediate_score, step)``.
            If absent, Hyperband degrades gracefully to a fixed-fidelity
            study with TPE sampling.
        max_resource: Total budget per trial (e.g. epochs). Used as the
            upper bound of intermediate steps.
        reduction_factor: Successive-halving factor (default 3).
    """

    def __init__(
        self,
        model_fn: Callable[[dict], torch.nn.Module],
        param_space: dict[str, tuple[Any, ...] | list[Any]],
        train_fn: Callable[..., float],
        n_trials: int = 50,
        device: str = "cpu",
        verbose: bool = True,
        seed: int | None = None,
        max_resource: int = 27,
        reduction_factor: int = 3,
    ) -> None:
        super().__init__(
            model_fn=model_fn,
            param_space=param_space,
            train_fn=train_fn,
            n_trials=n_trials,
            device=device,
            verbose=verbose,
            seed=seed,
        )
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor

    def search(self) -> HPOResult:
        _require_optuna()
        import optuna

        history: list[tuple[int, float, dict]] = []

        def objective(trial: "optuna.Trial") -> float:
            params = _suggest_params(trial, self.param_space)
            model = self.model_fn(params).to(self.device)

            def report_callback(score: float, step: int) -> None:
                trial.report(float(score), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                final = self.train_fn(model, params, report_callback=report_callback)
            except TypeError:
                # train_fn doesn't accept report_callback — fall back to single-fidelity.
                final = self.train_fn(model, params)

            score = float(final)
            history.append((trial.number, score, params))
            return score

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=self.max_resource,
            reduction_factor=self.reduction_factor,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
            catch=(optuna.TrialPruned,),
        )

        best = study.best_trial
        return HPOResult(
            best_params=dict(best.params),
            best_score=float(best.value),
            history=history,
        )

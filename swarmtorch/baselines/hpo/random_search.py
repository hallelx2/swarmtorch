"""Random search baseline — the floor any HPO method should beat."""

import random
from typing import Any

from swarmtorch.baselines.hpo.base import BaselineHPO, HPOResult


class RandomSearchBaseline(BaselineHPO):
    """Uniform random sampling over the search space.

    The first searcher to add to any HPO comparison: a sophisticated
    method that fails to beat random search isn't doing useful Bayesian
    inference (Bergstra & Bengio 2012).
    """

    def search(self) -> HPOResult:
        rng = random.Random(self.seed)
        best_params: dict | None = None
        best_score = float("inf")
        history: list[tuple[int, float, dict]] = []

        for i in range(self.n_trials):
            params = self._sample(rng)
            score = self._evaluate(params)
            history.append((i, score, params))
            if score < best_score:
                best_score = score
                best_params = params
            if self.verbose:
                print(f"[RandomSearch trial {i}] score={score:.4f} params={params}")

        assert best_params is not None
        return HPOResult(best_params=best_params, best_score=best_score, history=history)

    def _sample(self, rng: random.Random) -> dict:
        params: dict[str, Any] = {}
        for key, value in self.param_space.items():
            if isinstance(value, list):
                params[key] = rng.choice(value)
            else:
                lo, hi = float(value[0]), float(value[1])
                params[key] = rng.uniform(lo, hi)
        return params

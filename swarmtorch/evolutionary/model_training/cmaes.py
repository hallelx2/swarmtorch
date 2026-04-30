"""CMA-ES wrapped as a swarmtorch optimizer.

The Covariance Matrix Adaptation Evolution Strategy (Hansen 2003) is the
strongest gradient-free black-box optimizer for moderate-dimensional
non-convex problems and is the natural baseline for any benchmark that
includes swarm/evolutionary methods. This module wraps the reference
implementation from `pycma` so we get the canonical algorithm without
reimplementing the rank-mu / rank-1 covariance updates ourselves.

Install pycma to use this optimizer::

    pip install cma
"""

from typing import Any

import numpy as np
import torch

from swarmtorch.base import SwarmOptimizer


class CMAES(SwarmOptimizer):
    """Covariance Matrix Adaptation Evolution Strategy optimizer.

    Args:
        params: Model parameters to optimize.
        swarm_size: Population size lambda (default: pycma's heuristic
            ``4 + floor(3 * ln(d))``). When set explicitly it is forwarded
            as ``popsize`` to ``cma.CMAEvolutionStrategy``.
        sigma0: Initial step size (default: ``init_sigma * std(weights)``).
        device: Device to run computations on. Note that pycma operates on
            numpy arrays internally, so for very large parameter vectors
            CPU<->GPU transfers may dominate.
        init_strategy: Inherited from ``SwarmOptimizer`` — the mean of the
            CMA-ES distribution is initialized from the model's current
            (Kaiming/Xavier-initialized) weights when ``"model"``.

    Example:
        >>> import torch
        >>> from swarmtorch.evolutionary.model_training.cmaes import CMAES
        >>> model = torch.nn.Linear(10, 2)
        >>> opt = CMAES(model.parameters(), swarm_size=20)
        >>> for _ in range(10):
        ...     def closure():
        ...         return torch.nn.functional.mse_loss(model(x), y)
        ...     opt.step(closure)
    """

    def __init__(
        self,
        params: Any,
        swarm_size: int | None = None,
        sigma0: float | None = None,
        device: str = "cpu",
        init_strategy: str = "model",
        init_sigma: float = 0.1,
        **kwargs: Any,
    ) -> None:
        # CMA-ES picks its own default popsize if ``swarm_size`` is None.
        # We forward a sentinel of 0 to ``defaults`` so the registry shape
        # stays consistent and resolve the real value in ``_init_swarm``.
        forwarded_size = swarm_size if swarm_size is not None else 0
        super().__init__(
            params,
            swarm_size=forwarded_size,
            device=device,
            init_strategy=init_strategy,
            init_sigma=init_sigma,
        )
        self._user_swarm_size = swarm_size
        self.sigma0 = sigma0
        self.iteration_count = 0
        self.best_position: torch.Tensor | None = None
        self.best_fitness = torch.tensor(float("inf"), device=self.device)
        self._es: Any = None  # cma.CMAEvolutionStrategy, lazily created

    def _init_swarm(self) -> None:
        try:
            import cma
        except ImportError as e:  # pragma: no cover - exercised only without pycma
            raise ImportError(
                "CMA-ES requires the 'cma' package. Install it with: pip install cma"
            ) from e

        param_shape = self._get_param_shape()
        d = int(param_shape[0])

        x0 = self._get_params().detach().cpu().double().numpy()

        # Default sigma scales with the spread of the model's current weights.
        if self.sigma0 is None:
            std = float(self._get_params().std().clamp(min=1e-3).item())
            sigma0 = std * self.init_sigma
        else:
            sigma0 = float(self.sigma0)

        opts: dict[str, Any] = {"verbose": -9}
        if self._user_swarm_size is not None and self._user_swarm_size > 0:
            opts["popsize"] = int(self._user_swarm_size)

        self._es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        # Resolved population size (pycma may pick its own default).
        self.swarm_size = int(self._es.popsize)
        self.defaults["swarm_size"] = self.swarm_size

        # Allocate positions buffer so `_swarm_state` and the rest of the
        # SwarmOptimizer machinery have something coherent to serialize.
        self.positions = torch.empty(self.swarm_size, d, device=self.device)
        self.fitness = torch.full(
            (self.swarm_size,), float("inf"), device=self.device
        )
        self.best_position = self._get_params().detach().clone()

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        es = self._es
        if es is None:
            return

        # Ask CMA-ES for the next population.
        candidates = es.ask()  # list[ndarray] of length popsize
        positions = torch.from_numpy(np.asarray(candidates, dtype=np.float32)).to(
            self.device
        )
        self.positions = positions

        # Evaluate every candidate. Uses the vectorized vmap path when a
        # functional closure has been registered; otherwise the plain
        # closure loop.
        fitness = self._evaluate_fitness(self.positions, closure)
        self.fitness = fitness

        # Hand fitness values back to CMA-ES.
        es.tell(candidates, fitness.detach().cpu().double().tolist())

        # Track best-of-history.
        best_idx = int(torch.argmin(fitness).item())
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx].detach().clone()
            self.best_position = self.positions[best_idx].detach().clone()

        # Surface the running best as the model's current weights so the
        # user can call `model(...)` between steps and see good predictions.
        if self.best_position is not None:
            self._set_params(self.best_position)
        self.iteration_count += 1

    def _swarm_state(self) -> dict:
        state = super()._swarm_state()
        # pycma's internal state (covariance matrix, paths) is not trivially
        # picklable here; resuming a CMA-ES run mid-flight requires
        # CMAEvolutionStrategy.pickle/unpickle. For now we persist the
        # best-so-far so users can resume model weights at minimum.
        state["iteration_count"] = self.iteration_count
        return state

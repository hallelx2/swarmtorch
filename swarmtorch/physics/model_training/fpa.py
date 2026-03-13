from typing import Any
import torch
import math
from swarmtorch.base import SwarmOptimizer


class FPA(SwarmOptimizer):
    """Flower Pollination Algorithm (FPA) optimizer for PyTorch models."""

    def __init__(
        self,
        params: Any,
        swarm_size: int = 30,
        p: float = 0.8,
        device: str = "cpu",
    ) -> None:
        super().__init__(params, swarm_size=swarm_size, device=device, p=p)
        self.p = p
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.fitness = torch.full((self.swarm_size,), float("inf"), device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _levy_flight(self, dim: int) -> torch.Tensor:
        beta = 1.5
        sigma = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = torch.randn(dim, device=self.device) * sigma
        v = torch.randn(dim, device=self.device)
        step = u / (v.abs() ** (1 / beta))
        return step

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.positions, closure)
        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()

        for i in range(self.swarm_size):
            if torch.rand(1, device=self.device).item() < self.p:
                self.positions[i] = self.best_position + self._levy_flight(
                    self.positions.shape[1]
                ) * (self.positions[i] - self.best_position)
            else:
                j, k = torch.randint(0, self.swarm_size, (2,)).tolist()
                self.positions[i] = self.positions[i] + torch.rand_like(
                    self.positions[i]
                ) * (self.positions[j] - self.positions[k])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class GSA(SwarmOptimizer):
    """Gravitational Search Algorithm (GSA) optimizer for PyTorch models."""

    def __init__(
        self,
        params: Any,
        swarm_size: int = 30,
        g0: float = 100.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(params, swarm_size=swarm_size, device=device, g0=g0)
        self.g0 = g0
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]

        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.velocities = torch.zeros_like(self.positions)
        self.fitness = torch.full((self.swarm_size,), float("inf"), device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _set_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.data.copy_(flat_params[idx : idx + numel].reshape(p.shape))
                idx += numel

    def _get_params(self) -> torch.Tensor:
        params = []
        for group in self.param_groups:
            for p in group["params"]:
                params.append(p.data.flatten())
        return torch.cat(params)

    def _evaluate_fitness(
        self, particles: torch.Tensor, closure: Any = None
    ) -> torch.Tensor:
        if closure is None:
            raise ValueError("GSA requires a closure function")

        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            loss = closure()
            fitness[i] = loss.detach()
        return fitness

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.positions, closure)

        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()

        g = self.g0 / (self.iteration_count + 1)

        masses = (fitness - torch.max(fitness)) / (
            torch.min(fitness) - torch.max(fitness) + 1e-10
        )
        masses = torch.exp(masses)
        masses = masses / (torch.sum(masses) + 1e-10)

        for i in range(self.swarm_size):
            force = torch.zeros_like(self.positions[i])
            for j in range(self.swarm_size):
                if i != j:
                    dist = torch.norm(self.positions[j] - self.positions[i]) + 1e-10
                    force += (
                        torch.rand_like(self.positions[i])
                        * g
                        * masses[j]
                        * (self.positions[j] - self.positions[i])
                        / dist
                    )

            self.velocities[i] = (
                torch.rand_like(self.velocities[i]) * self.velocities[i] + force
            )
            self.positions[i] = self.positions[i] + self.velocities[i]

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

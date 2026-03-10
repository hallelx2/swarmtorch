from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class CEM(SwarmOptimizer):
    """Cross-Entropy Method (CEM) optimizer for PyTorch models."""

    def __init__(
        self, params: Any, population_size: int = 30, device: str = "cpu"
    ) -> None:
        super().__init__(params, swarm_size=population_size, device=device)
        self.population_size = population_size
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.population_size = self.defaults["swarm_size"]
        self.positions = torch.zeros(
            self.population_size, param_shape[0], device=self.device
        )
        self.samples = torch.zeros(
            self.population_size, param_shape[0], device=self.device
        )
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)
        self.mean = torch.zeros(param_shape[0], device=self.device)
        self.std = torch.ones(param_shape[0], device=self.device)

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        self.samples = self.mean + self.std * torch.randn(
            self.population_size, self.mean.shape[0], device=self.device
        )
        fitness = self._evaluate_fitness(self.samples, closure)

        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.samples[best_idx].clone()

        n_elite = self.population_size // 5
        elite_idx = torch.argsort(fitness)[:n_elite]
        elite_samples = self.samples[elite_idx]

        self.mean = torch.mean(elite_samples, dim=0)
        self.std = torch.std(elite_samples, dim=0) + 0.01

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class PFA(SwarmOptimizer):
    """Peacock Algorithm (PFA) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

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
            self.positions[i] = self.best_position + torch.randn_like(
                self.positions[i]
            ) * (1 - self.iteration_count / 1000)

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class ARS(SwarmOptimizer):
    """Archimedes Optimization Algorithm (ARS) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.velocities = torch.zeros_like(self.positions)
        self.density = torch.rand(self.swarm_size, device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return
        fitness = self._evaluate_fitness(self.positions, closure)

        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()

        min_fit = torch.min(fitness)
        max_fit = torch.max(fitness)
        self.density = (fitness - min_fit) / (max_fit - min_fit + 1e-10)

        for i in range(self.swarm_size):
            force = torch.zeros_like(self.positions[i])
            for j in range(self.swarm_size):
                if i != j:
                    force += (self.positions[j] - self.positions[i]) * (
                        self.density[j] - self.density[i]
                    )

            self.velocities[i] = self.velocities[i] * 0.5 + force * 0.1
            self.positions[i] = self.positions[i] + self.velocities[i]

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class FDA(SwarmOptimizer):
    """Forest Defense Algorithm (FDA) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

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
            center = torch.mean(self.positions, dim=0)
            self.positions[i] = center + torch.randn_like(self.positions[i]) * (
                1 - self.iteration_count / 1000
            )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class CA(SwarmOptimizer):
    """Cultural Algorithm (CA) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)
        self.belief_space = torch.zeros(param_shape[0], device=self.device)

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return
        fitness = self._evaluate_fitness(self.positions, closure)
        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()
            self.belief_space = self.best_position.clone()

        for i in range(self.swarm_size):
            self.positions[i] = (
                self.belief_space + torch.randn_like(self.positions[i]) * 0.5
            )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class PBIL(SwarmOptimizer):
    """Population-Based Incremental Learning (PBIL) optimizer for PyTorch models."""

    def __init__(
        self, params: Any, population_size: int = 30, device: str = "cpu"
    ) -> None:
        super().__init__(params, swarm_size=population_size, device=device)
        self.population_size = population_size
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.population_size = self.defaults["swarm_size"]
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)
        self.probability = torch.ones(param_shape[0], device=self.device) * 0.5

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        samples = (
            torch.rand(
                self.population_size, self.probability.shape[0], device=self.device
            )
            < self.probability
        ).float()
        fitness = self._evaluate_fitness(samples, closure)

        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = samples[best_idx].clone()

        self.probability = 0.1 * self.best_position + 0.9 * self.probability

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

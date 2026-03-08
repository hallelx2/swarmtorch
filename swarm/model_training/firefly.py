from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class Firefly(SwarmOptimizer):
    """Firefly Algorithm (FA) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, alpha: float = 0.5, beta0: float = 1.0, gamma: float = 1.0, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device, alpha=alpha, beta0=beta0, gamma=gamma)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _set_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(flat_params[idx : idx + p.numel()].reshape(p.shape))
                idx += p.numel()

    def _get_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for group in self.param_groups for p in group["params"]])

    def _evaluate_fitness(self, particles: torch.Tensor, closure: Any = None) -> torch.Tensor:
        if closure is None:
            raise ValueError("FA requires a closure function")
        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            fitness[i] = closure().detach()
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

        for i in range(self.swarm_size):
            for j in range(self.swarm_size):
                if fitness[j] < fitness[i]:
                    dist = torch.norm(self.positions[i] - self.positions[j])
                    beta = self.beta0 * torch.exp(-self.gamma * dist ** 2)
                    self.positions[i] = self.positions[i] + beta * (self.positions[j] - self.positions[i]) + self.alpha * (torch.rand_like(self.positions[i]) - 0.5)

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Bat(SwarmOptimizer):
    """Bat Algorithm (BA) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.velocities = torch.zeros_like(self.positions)
        self.frequencies = torch.zeros(self.swarm_size, device=self.device)
        self.loudnesses = torch.full((self.swarm_size,), 0.5, device=self.device)
        self.pulse_rates = torch.full((self.swarm_size,), 0.5, device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _set_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(flat_params[idx : idx + p.numel()].reshape(p.shape))
                idx += p.numel()

    def _get_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for group in self.param_groups for p in group["params"]])

    def _evaluate_fitness(self, particles: torch.Tensor, closure: Any = None) -> torch.Tensor:
        if closure is None:
            raise ValueError("BA requires a closure function")
        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            fitness[i] = closure().detach()
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

        for i in range(self.swarm_size):
            self.frequencies[i] = 0 + 2 * torch.rand(1, device=self.device).item()
            self.velocities[i] += (self.positions[i] - self.best_position) * self.frequencies[i]
            self.positions[i] = self.positions[i] + self.velocities[i]

            if torch.rand(1, device=self.device).item() > self.pulse_rates[i]:
                self.positions[i] = self.best_position + 0.001 * torch.randn_like(self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Dragonfly(SwarmOptimizer):
    """Dragonfly Algorithm (DA) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _set_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(flat_params[idx : idx + p.numel()].reshape(p.shape))
                idx += p.numel()

    def _get_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for group in self.param_groups for p in group["params"]])

    def _evaluate_fitness(self, particles: torch.Tensor, closure: Any = None) -> torch.Tensor:
        if closure is None:
            raise ValueError("DA requires a closure function")
        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            fitness[i] = closure().detach()
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

        for i in range(self.swarm_size):
            neighbor_idx = torch.randint(0, self.swarm_size, (3,)).tolist()
            separation = -torch.sum(self.positions[i] - self.positions[neighbor_idx[0]])
            alignment = torch.mean(self.positions[neighbor_idx[1]]) - self.positions[i]
            cohesion = torch.mean(self.positions[neighbor_idx[2]]) - self.positions[i]
            
            self.positions[i] += (separation + alignment + cohesion) * 0.1 + (self.best_position - self.positions[i]) * 0.01

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

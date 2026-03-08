from typing import Any
import torch
import math
from swarmtorch.base import SwarmOptimizer


class CuckooSearch(SwarmOptimizer):
    """Cuckoo Search (CS) optimizer for PyTorch models."""

    def __init__(
        self, params: Any, swarm_size: int = 30, pa: float = 0.25, device: str = "cpu"
    ) -> None:
        super().__init__(params, swarm_size=swarm_size, device=device, pa=pa)
        self.pa = pa
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
        return torch.cat(
            [p.data.flatten() for group in self.param_groups for p in group["params"]]
        )

    def _evaluate_fitness(
        self, particles: torch.Tensor, closure: Any = None
    ) -> torch.Tensor:
        if closure is None:
            raise ValueError("CS requires a closure function")
        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            fitness[i] = closure().detach()
        return fitness

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
            if torch.rand(1, device=self.device).item() > self.pa:
                self.positions[i] = self.positions[i] + 0.01 * self._levy_flight(
                    self.positions.shape[1]
                ) * (self.positions[i] - self.best_position)

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Salp(SwarmOptimizer):
    """Salp Chain Algorithm (Salp) optimizer for PyTorch models."""

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
        return torch.cat(
            [p.data.flatten() for group in self.param_groups for p in group["params"]]
        )

    def _evaluate_fitness(
        self, particles: torch.Tensor, closure: Any = None
    ) -> torch.Tensor:
        if closure is None:
            raise ValueError("Salp requires a closure function")
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

        c1 = 2 * math.exp(-((4 * self.iteration_count / 1000) ** 2))

        for i in range(self.swarm_size):
            if i == 0:
                self.positions[i] = self.best_position + c1 * (
                    torch.rand(self.positions.shape[1], device=self.device) * 2 - 1
                )
            else:
                self.positions[i] = (
                    self.positions[i] + self.positions[i - 1]
                ) / 2 + c1 * (
                    torch.rand(self.positions.shape[1], device=self.device) * 2 - 1
                )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Bee(SwarmOptimizer):
    """Artificial Bee Colony (ABC/Bee) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.trials = torch.zeros(self.swarm_size, device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _set_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                p.data.copy_(flat_params[idx : idx + p.numel()].reshape(p.shape))
                idx += p.numel()

    def _get_params(self) -> torch.Tensor:
        return torch.cat(
            [p.data.flatten() for group in self.param_groups for p in group["params"]]
        )

    def _evaluate_fitness(
        self, particles: torch.Tensor, closure: Any = None
    ) -> torch.Tensor:
        if closure is None:
            raise ValueError("Bee requires a closure function")
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
            j = torch.randint(0, self.positions.shape[1], (1,)).item()
            k = torch.randint(0, self.swarm_size, (1,)).item()
            while k == i:
                k = torch.randint(0, self.swarm_size, (1,)).item()

            new_pos = self.positions[i].clone()
            new_pos[j] = self.positions[i, j] + (
                torch.rand(1, device=self.device).item() * 2 - 1
            ) * (self.positions[i, j] - self.positions[k, j])

            self._set_params(new_pos)
            new_fitness = closure().detach()

            if new_fitness < fitness[i]:
                self.positions[i] = new_pos
                self.trials[i] = 0
            else:
                self.trials[i] += 1

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Fish(SwarmOptimizer):
    """Artificial Fish Swarm (AFS/Fish) optimizer for PyTorch models."""

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
        return torch.cat(
            [p.data.flatten() for group in self.param_groups for p in group["params"]]
        )

    def _evaluate_fitness(
        self, particles: torch.Tensor, closure: Any = None
    ) -> torch.Tensor:
        if closure is None:
            raise ValueError("Fish requires a closure function")
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
            center = torch.mean(self.positions, dim=0)
            self._set_params(center)
            center_fitness = closure().detach()

            if center_fitness < fitness[i]:
                self.positions[i] = center + torch.rand_like(
                    self.positions[i]
                ) * 0.5 * (center - self.positions[i])
            else:
                partner_idx = torch.randint(0, self.swarm_size, (1,)).item()
                self.positions[i] = self.positions[i] + torch.rand_like(
                    self.positions[i]
                ) * (self.positions[partner_idx] - self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

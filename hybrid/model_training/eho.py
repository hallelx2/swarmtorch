from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class EHO(SwarmOptimizer):
    """Elephant Herding Optimization (EHO) optimizer for PyTorch models."""

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
            raise ValueError("EHO requires a closure function")
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

        cl = 0.5
        for i in range(1, self.swarm_size):
            self.positions[i] = (
                cl * self.positions[i] + (1 - cl) * self.positions[best_idx]
            )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class ChickenSwarm(SwarmOptimizer):
    """Chicken Swarm Optimization (CSO) optimizer for PyTorch models."""

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
            raise ValueError("CSO requires a closure function")
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

        num_roosters = max(1, self.swarm_size // 10)
        for i in range(self.swarm_size):
            if i < num_roosters:
                self.positions[i] = (
                    self.positions[i] + torch.randn_like(self.positions[i]) * 0.5
                )
            else:
                mother_idx = torch.randint(0, num_roosters, (1,)).item()
                self.positions[i] = self.positions[i] + torch.rand_like(
                    self.positions[i]
                ) * (self.positions[mother_idx] - self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class SMA(SwarmOptimizer):
    """Slime Mold Algorithm (SMA) optimizer for PyTorch models."""

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
            raise ValueError("SMA requires a closure function")
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
        sorted_idx = torch.argsort(fitness)
        if fitness[sorted_idx[0]] < self.best_fitness:
            self.best_fitness = fitness[sorted_idx[0]]
            self.best_position = self.positions[sorted_idx[0]].clone()

        w = torch.exp(
            torch.arange(self.swarm_size, device=self.device).float()
            / (-self.swarm_size)
        )

        for i in range(self.swarm_size):
            p = torch.tanh(torch.abs(fitness[i]))
            vb = torch.rand(self.positions.shape[1], device=self.device) * 2 - 1
            vc = torch.rand(self.positions.shape[1], device=self.device) * 2 - 1

            idx1 = torch.randint(0, self.swarm_size, (1,)).item()
            idx2 = torch.randint(0, self.swarm_size, (1,)).item()

            self.positions[i] = (
                self.best_position
                + vb * p * (self.positions[idx1] - self.positions[idx2])
                + vc * w[i] * (self.best_position - self.positions[i])
            )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class AFSA(SwarmOptimizer):
    """Artificial Fish Swarm Algorithm (AFSA) optimizer for PyTorch models."""

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

        visual = 0.5
        step = 0.3
        for i in range(self.swarm_size):
            center = torch.mean(self.positions, dim=0)
            self._set_params(center)
            center_fit = closure().detach()

            if center_fit < fitness[i]:
                self.positions[i] = self.positions[i] + step * (
                    center - self.positions[i]
                ) / (torch.norm(center - self.positions[i]) + 1e-10)
            else:
                partner = torch.randint(0, self.swarm_size, (1,)).item()
                dist = torch.norm(self.positions[i] - self.positions[partner])
                if dist < visual:
                    self.positions[i] = self.positions[i] + step * (
                        self.positions[partner] - self.positions[i]
                    ) / (dist + 1e-10)
                else:
                    self.positions[i] = (
                        self.positions[i] + torch.rand_like(self.positions[i]) * visual
                    )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class HSA(SwarmOptimizer):
    """Harmony Search Algorithm (HSA) optimizer for PyTorch models."""

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
            if torch.rand(1, device=self.device).item() < 0.9:
                mem_idx = torch.randint(0, self.swarm_size, (1,)).item()
                new_pos = self.positions[mem_idx].clone()
                if torch.rand(1, device=self.device).item() < 0.3:
                    new_pos += (torch.rand_like(new_pos) - 0.5) * 0.2
                self.positions[i] = new_pos
            else:
                self.positions[i] = torch.rand_like(self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class DFO2(SwarmOptimizer):
    """Dingo Optimization Algorithm (DFO2) - variant for PyTorch models."""

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
            explorer = torch.randint(0, self.swarm_size // 3, (1,)).item()
            follower = torch.randint(self.swarm_size // 3, self.swarm_size, (1,)).item()
            self.positions[i] = self.positions[explorer] + torch.rand_like(
                self.positions[i]
            ) * (self.positions[follower] - self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

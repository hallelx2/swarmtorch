from typing import Any
import torch
import math
import math
from swarmtorch.base import SwarmOptimizer


class SineCosine(SwarmOptimizer):
    """Sine Cosine Algorithm (SCA) optimizer for PyTorch models."""

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
            raise ValueError("SCA requires a closure function")
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

        max_iter = 1000
        a = 2
        r1 = a - self.iteration_count * (a / max_iter)
        
        for i in range(self.swarm_size):
            for j in range(self.positions.shape[1]):
                r2 = torch.rand(1, device=self.device).item() * 2 * 3.14159
                r3 = torch.rand(1, device=self.device).item() * 2
                r4 = torch.rand(1, device=self.device).item()
                
                if r4 < 0.5:
                    self.positions[i, j] = self.positions[i, j] + r1 * math.sin(r2) * torch.abs(r3 * self.best_position[j] - self.positions[i, j])
                else:
                    self.positions[i, j] = self.positions[i, j] + r1 * math.cos(r2) * torch.abs(r3 * self.best_position[j] - self.positions[i, j])
        
        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class MFO(SwarmOptimizer):
    """Moth Flame Algorithm (MFO) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.flames = torch.rand(self.swarm_size // 3, param_shape[0], device=self.device)
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
            raise ValueError("MFO requires a closure function")
        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            fitness[i] = closure().detach()
        return fitness

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return
        
        all_pos = torch.cat([self.positions, self.flames], dim=0)
        fitness = self._evaluate_fitness(all_pos, closure)
        
        sorted_idx = torch.argsort(fitness)
        self.best_fitness = fitness[sorted_idx[0]]
        self.best_position = all_pos[sorted_idx[0]].clone()
        
        self.positions = all_pos[:self.swarm_size].clone()
        self.flames = all_pos[self.swarm_size:].clone()
        
        max_iter = 1000
        t = (self.iteration_count / max_iter) * 2 - 1
        
        for i in range(self.swarm_size):
            for j in range(self.positions.shape[1]):
                flame_idx = int(i * self.flames.shape[0] / self.swarm_size)
                distance = torch.abs(self.positions[i, j] - self.flames[flame_idx, j])
                b = 1
                t_val = (t * (1 - j / self.positions.shape[1]))
                self.positions[i, j] = distance * math.exp(b * t_val) * math.cos(2 * 3.14159 * t_val) + self.flames[flame_idx, j]
        
        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

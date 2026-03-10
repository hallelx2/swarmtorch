from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class TLBO(SwarmOptimizer):
    """Teaching-Learning-Based Optimization (TLBO) optimizer for PyTorch models."""

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

        teacher = self.positions[best_idx].clone()
        mean_pos = torch.mean(self.positions, dim=0)
        tf = torch.randint(1, 3, (1,)).item()

        for i in range(self.swarm_size):
            new_pos = self.positions[i] + torch.rand_like(self.positions[i]) * (
                teacher - tf * mean_pos
            )
            self.positions[i] = new_pos

            partner_idx = torch.randint(0, self.swarm_size, (1,)).item()
            if fitness[i] < fitness[partner_idx]:
                self.positions[i] += torch.rand_like(self.positions[i]) * (
                    self.positions[i] - self.positions[partner_idx]
                )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class HarmonySearch(SwarmOptimizer):
    """Harmony Search (HS) optimizer for PyTorch models."""

    def __init__(
        self,
        params: Any,
        swarm_size: int = 30,
        hmcr: float = 0.9,
        par: float = 0.3,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            params, swarm_size=swarm_size, device=device, hmcr=hmcr, par=par
        )
        self.hmcr = hmcr
        self.par = par
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
            if torch.rand(1, device=self.device).item() < self.hmcr:
                selected_idx = torch.randint(0, self.swarm_size, (1,)).item()
                new_pos = self.positions[selected_idx].clone()
                if torch.rand(1, device=self.device).item() < self.par:
                    new_pos += torch.randn_like(new_pos) * 0.1
                self.positions[i] = new_pos
            else:
                self.positions[i] = torch.rand(
                    self.positions.shape[1], device=self.device
                )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

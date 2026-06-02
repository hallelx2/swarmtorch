from typing import Any
import torch
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
        self.positions = self._init_positions(param_shape[0])
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

        max_iter = 1000
        a = 2
        r1 = a - self.iteration_count * (a / max_iter)

        # Vectorized SCA update (formerly a per-element Python double loop,
        # which forced millions of GPU<->CPU syncs and dominated wall-clock).
        # r2 in [0, 2*pi), r3 in [0, 2), r4 in [0, 1) per element.
        r2 = torch.rand_like(self.positions) * (2 * math.pi)
        r3 = torch.rand_like(self.positions) * 2
        r4 = torch.rand_like(self.positions)

        target = torch.abs(r3 * self.best_position.unsqueeze(0) - self.positions)
        trig = torch.where(r4 < 0.5, torch.sin(r2), torch.cos(r2))
        self.positions = self.positions + r1 * trig * target

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
        self.positions = self._init_positions(param_shape[0])
        self.flames = self._init_positions(param_shape[0])[: self.swarm_size // 3].clone()
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        all_pos = torch.cat([self.positions, self.flames], dim=0)
        fitness = self._evaluate_fitness(all_pos, closure)

        sorted_idx = torch.argsort(fitness)
        self.best_fitness = fitness[sorted_idx[0]]
        self.best_position = all_pos[sorted_idx[0]].clone()

        self.positions = all_pos[: self.swarm_size].clone()
        self.flames = all_pos[self.swarm_size :].clone()

        max_iter = 1000
        t = (self.iteration_count / max_iter) * 2 - 1
        b = 1.0
        d = self.positions.shape[1]
        n_flames = self.flames.shape[0]

        # Vectorized MFO update (was a per-element Python double loop).
        # Each particle i spirals around flame floor(i * n_flames / swarm).
        flame_idx = (
            torch.arange(self.swarm_size, device=self.device) * n_flames
            // self.swarm_size
        ).clamp(max=n_flames - 1)
        chosen = self.flames[flame_idx]  # (swarm_size, d)

        # t_val decays across dimensions: t * (1 - j/d), broadcast over particles.
        j = torch.arange(d, device=self.device, dtype=self.positions.dtype)
        t_val = (t * (1 - j / d)).unsqueeze(0)  # (1, d)

        distance = torch.abs(self.positions - chosen)
        self.positions = (
            distance * torch.exp(b * t_val) * torch.cos(2 * math.pi * t_val) + chosen
        )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

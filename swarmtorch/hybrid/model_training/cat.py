from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class CatSwarm(SwarmOptimizer):
    """Cat Swarm Optimization (CSO/CatSwarm) optimizer for PyTorch models."""

    def __init__(
        self, params: Any, swarm_size: int = 30, smp: int = 5, device: str = "cpu"
    ) -> None:
        super().__init__(params, swarm_size=swarm_size, device=device, smp=smp)
        self.smp = smp
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.velocities = torch.zeros_like(self.positions)
        self.flags = torch.randint(0, 2, (self.swarm_size,), device=self.device)
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
            if self.flags[i] == 0:
                dim = self.positions.shape[1]
                idx = torch.randint(0, dim, (self.smp,)).tolist()
                candidate = self.positions[i].clone()
                for j in idx:
                    candidate[j] = self.best_position[j]
                self._set_params(candidate)
                if closure().detach() < fitness[i]:
                    self.positions[i] = candidate
            else:
                self.positions[i] = self.positions[i] + self.velocities[i]

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Cockroach(SwarmOptimizer):
    """Cockroach Swarm Optimization (CSO/Cockroach) optimizer for PyTorch models."""

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
            j = torch.randint(0, self.swarm_size, (1,)).item()
            k = torch.randint(0, self.swarm_size, (1,)).item()
            self.positions[i] = (
                self.positions[i]
                + 0.3 * (self.best_position - self.positions[i])
                + 0.1 * (self.positions[j] - self.positions[k])
            )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Coati(SwarmOptimizer):
    """Coati Optimization Algorithm (COA/Coati) optimizer for PyTorch models."""

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

        max_iter = 1000
        t = self.iteration_count / max_iter

        for i in range(self.swarm_size):
            if t < 0.5:
                j = torch.randint(0, self.swarm_size, (1,)).item()
                self.positions[i] = self.positions[i] + torch.rand_like(
                    self.positions[i]
                ) * (self.positions[j] - self.positions[i])
            else:
                self.positions[i] = (
                    self.best_position
                    + (torch.rand(self.positions.shape[1], device=self.device) * 2 - 1)
                    * t
                )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class GorillaTroopsOptimizer(SwarmOptimizer):
    """Gorilla Troops Optimization (GTO) optimizer for PyTorch models."""

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

        p = 0.03
        g = 0.5
        for i in range(self.swarm_size):
            if torch.rand(1, device=self.device).item() < p:
                self.positions[i] = g * torch.rand(
                    self.positions.shape[1], device=self.device
                )
            else:
                c1 = torch.rand(1, device=self.device).item()
                c2 = torch.rand(1, device=self.device).item()
                if c1 > 0.5:
                    self.positions[i] = self.best_position + c2 * (
                        torch.rand(self.positions.shape[1], device=self.device) * 2 - 1
                    )
                else:
                    self.positions[i] = self.positions[i] - c2 * (
                        self.best_position - self.positions[i]
                    )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Gorilla(GorillaTroopsOptimizer):
    """Gorilla Troops Optimization (GTO/Gorilla) optimizer for PyTorch models."""

    pass


class JSO(SwarmOptimizer):
    """Jaya Optimization (JSO) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.worst_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)
        self.worst_fitness = torch.tensor(-float("inf"), device=self.device)

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return
        fitness = self._evaluate_fitness(self.positions, closure)

        best_idx = torch.argmin(fitness)
        worst_idx = torch.argmax(fitness)

        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()
        if fitness[worst_idx] > self.worst_fitness:
            self.worst_fitness = fitness[worst_idx]
            self.worst_position = self.positions[worst_idx].clone()

        for i in range(self.swarm_size):
            r1 = torch.rand_like(self.positions[i])
            r2 = torch.rand_like(self.positions[i])
            self.positions[i] = (
                self.positions[i]
                + r1 * (self.best_position - self.positions[i])
                - r2 * (self.worst_position - self.positions[i])
            )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class KHA(SwarmOptimizer):
    """Krill Herd Algorithm (KHA) optimizer for PyTorch models."""

    def __init__(self, params: Any, swarm_size: int = 30, device: str = "cpu") -> None:
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]
        self.positions = torch.rand(self.swarm_size, param_shape[0], device=self.device)
        self.velocities = torch.zeros_like(self.positions)
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
            target_idx = torch.randint(0, self.swarm_size, (1,)).item()

            self.velocities[i] = (
                0.1 * self.velocities[i]
                + 0.5
                * torch.rand_like(self.velocities[i])
                * (self.best_position - self.positions[i])
                + 0.5
                * torch.rand_like(self.velocities[i])
                * (center - self.positions[i])
                + 0.5
                * torch.rand_like(self.velocities[i])
                * (self.positions[target_idx] - self.positions[i])
            )

            self.positions[i] = self.positions[i] + self.velocities[i]

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

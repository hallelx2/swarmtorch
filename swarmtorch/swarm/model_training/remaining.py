from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class DFO(SwarmOptimizer):
    """Dingo Optimization Algorithm (DFO) optimizer for PyTorch models."""

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
            raise ValueError("DFO requires a closure function")
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
            r = torch.rand_like(self.positions[i])
            self.positions[i] = (
                self.best_position
                + r * (self.positions[i] - self.best_position)
                + torch.randn_like(self.positions[i]) * 0.1
            )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class MBO(SwarmOptimizer):
    """Marriage in Honey Bees Optimization (MBO) optimizer for PyTorch models."""

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
            raise ValueError("MBO requires a closure function")
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
            queen_idx = torch.randint(0, self.swarm_size // 3, (1,)).item()
            self.positions[i] = self.positions[queen_idx] + torch.rand_like(
                self.positions[i]
            ) * (self.best_position - self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class CSA(SwarmOptimizer):
    """Crow Search Algorithm (CSA) optimizer for PyTorch models."""

    def __init__(
        self, params: Any, swarm_size: int = 30, ap: float = 0.1, device: str = "cpu"
    ) -> None:
        super().__init__(params, swarm_size=swarm_size, device=device, ap=ap)
        self.ap = ap
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
            raise ValueError("CSA requires a closure function")
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
            if torch.rand(1, device=self.device).item() > self.ap:
                j = torch.randint(0, self.swarm_size, (1,)).item()
                self.positions[i] = self.positions[i] + 2 * torch.rand_like(
                    self.positions[i]
                ) * (self.positions[j] - self.positions[i])
            else:
                self.positions[i] = torch.rand_like(self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class AOA(SwarmOptimizer):
    """Arithmetic Optimization Algorithm (AOA) optimizer for PyTorch models."""

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
            raise ValueError("AOA requires a closure function")
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
        mu = 0.5
        t = self.iteration_count / max_iter

        for i in range(self.swarm_size):
            for j in range(self.positions.shape[1]):
                r = torch.rand(1, device=self.device).item()
                if r < 0.5:
                    self.positions[i, j] = (
                        self.best_position[j]
                        / (mu + 1e-10)
                        * torch.exp(
                            torch.tensor(-t, dtype=torch.float32, device=self.device)
                        ).item()
                    )
                else:
                    self.positions[i, j] = (
                        self.best_position[j]
                        * torch.exp(
                            torch.tensor(t, dtype=torch.float32, device=self.device)
                        ).item()
                    )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class SOS(SwarmOptimizer):
    """Symbiotic Organisms Search (SOS) optimizer for PyTorch models."""

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
            raise ValueError("SOS requires a closure function")
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
            j = torch.randint(0, self.swarm_size, (1,)).item()
            while j == i:
                j = torch.randint(0, self.swarm_size, (1,)).item()

            r1 = torch.rand(1, device=self.device).item()
            torch.rand(1, device=self.device).item()

            mutualism = (
                (self.positions[i] + self.positions[j])
                / 2
                * (r1 * self.best_position - self.positions[i])
            )
            self.positions[i] = self.positions[i] + mutualism

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class DVBA(SwarmOptimizer):
    """Dwarf Mongoose Optimization (DVBA) optimizer for PyTorch models."""

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
            raise ValueError("DVBA requires a closure function")
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
            self.positions[i] = self.best_position + torch.randn_like(
                self.positions[i]
            ) * (1 - self.iteration_count / 1000)

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class ABCO(SwarmOptimizer):
    """Artificial Bee Colony Optimization (ABCO) optimizer for PyTorch models."""

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
            raise ValueError("ABCO requires a closure function")
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
            partner = torch.randint(0, self.swarm_size, (1,)).item()
            phi = torch.rand(1, device=self.device).item() * 2 - 1
            new_pos = self.positions[i] + phi * (
                self.positions[i] - self.positions[partner]
            )
            self._set_params(new_pos)
            if closure().detach() < fitness[i]:
                self.positions[i] = new_pos

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class GOA(SwarmOptimizer):
    """Grasshopper Optimization Algorithm (GOA) optimizer for PyTorch models."""

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
            raise ValueError("GOA requires a closure function")
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
        c = 1 - self.iteration_count / max_iter

        for i in range(self.swarm_size):
            s = torch.zeros_like(self.positions[i])
            for j in range(self.swarm_size):
                if i != j:
                    dist = torch.norm(self.positions[i] - self.positions[j])
                    s += (
                        (self.positions[j] - self.positions[i])
                        / (dist + 1e-10)
                        * torch.rand(1, device=self.device).item()
                    )

            self.positions[i] = c * s + self.best_position

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class HUS(SwarmOptimizer):
    """Husband and Wife Optimization (HUS) optimizer for PyTorch models."""

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
            raise ValueError("HUS requires a closure function")
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
            if i < self.swarm_size // 2:
                self.positions[i] = self.best_position + torch.rand_like(
                    self.positions[i]
                ) * (self.best_position - self.positions[i])
            else:
                partner = torch.randint(0, self.swarm_size // 2, (1,)).item()
                self.positions[i] = self.positions[i] + torch.rand_like(
                    self.positions[i]
                ) * (self.positions[partner] - self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class JY(SwarmOptimizer):
    """Jaya Algorithm (JY) optimizer for PyTorch models."""

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
        self.worst_fitness = torch.tensor(float("inf"), device=self.device)

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
            raise ValueError("JY requires a closure function")
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


class SPBO(SwarmOptimizer):
    """Sports Optimization (SPBO) optimizer for PyTorch models."""

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
            raise ValueError("SPBO requires a closure function")
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
            coach_idx = torch.randint(0, self.swarm_size // 3, (1,)).item()
            self.positions[i] = self.best_position + torch.rand_like(
                self.positions[i]
            ) * (self.positions[coach_idx] - self.positions[i])

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class RandomSearch(SwarmOptimizer):
    """Random Search (RandomSearch) optimizer for PyTorch models."""

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
            raise ValueError("RandomSearch requires a closure function")
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

        self.positions = torch.rand_like(self.positions)

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class IGWO(SwarmOptimizer):
    """Improved Grey Wolf Optimizer (IGWO/GWO-Improved) optimizer for PyTorch models."""

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
            raise ValueError("IGWO requires a closure function")
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

        max_iter = 1000
        a = 2 - self.iteration_count * (2 / max_iter)
        a2 = 2 - self.iteration_count * (2 / max_iter) * 0.5

        for i in range(self.swarm_size):
            for j in range(self.positions.shape[1]):
                r1 = torch.rand(1, device=self.device).item()
                r2 = torch.rand(1, device=self.device).item()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D1 = torch.abs(C1 * self.best_position[j] - self.positions[i, j])
                X1 = self.best_position[j] - A1 * D1

                r1 = torch.rand(1, device=self.device).item()
                r2 = torch.rand(1, device=self.device).item()

                a2 * (2 * r1 - 1)
                2 * r2
                X2 = self.best_position[j] + torch.randn(1, device=self.device).item()

                self.positions[i, j] = (X1 + X2) / 2

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class IWOA(SwarmOptimizer):
    """Improved Whale Optimization Algorithm (IWOA) optimizer for PyTorch models."""

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
            raise ValueError("IWOA requires a closure function")
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
        a = 2 - self.iteration_count * (2 / max_iter)
        l_val = torch.rand(1, device=self.device) * 2 - 1
        p = torch.rand(1, device=self.device).item()

        for i in range(self.swarm_size):
            if p < 0.5:
                r = torch.rand(1, device=self.device).item()
                if abs(a) < 1:
                    D = torch.abs(self.best_position - self.positions[i])
                    self.positions[i] = self.best_position - a * D * r
                else:
                    rand_idx = torch.randint(0, self.swarm_size, (1,)).item()
                    self.positions[i] = (
                        self.positions[rand_idx]
                        + torch.randn_like(self.positions[i]) * 0.1
                    )
            else:
                D = torch.abs(self.best_position - self.positions[i])
                self.positions[i] = (
                    D * torch.exp(l) * torch.cos(2 * 3.14159 * l) + self.best_position
                )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class ACGWO(SwarmOptimizer):
    """Accelerated Chaotic Grey Wolf Optimizer (ACGWO) optimizer for PyTorch models."""

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
            raise ValueError("ACGWO requires a closure function")
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

        alpha = self.positions[sorted_idx[0]]
        beta = self.positions[sorted_idx[1]] if len(sorted_idx) > 1 else alpha
        delta = self.positions[sorted_idx[2]] if len(sorted_idx) > 2 else alpha

        max_iter = 1000
        a = 2 - self.iteration_count * (2 / max_iter)

        for i in range(self.swarm_size):
            for j in range(self.positions.shape[1]):
                r1 = torch.rand(1, device=self.device).item()
                r2 = torch.rand(1, device=self.device).item()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D1 = torch.abs(C1 * alpha[j] - self.positions[i, j])
                X1 = alpha[j] - A1 * D1

                r1 = torch.rand(1, device=self.device).item()
                r2 = torch.rand(1, device=self.device).item()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D2 = torch.abs(C2 * beta[j] - self.positions[i, j])
                X2 = beta[j] - A2 * D2

                r1 = torch.rand(1, device=self.device).item()
                r2 = torch.rand(1, device=self.device).item()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D3 = torch.abs(C3 * delta[j] - self.positions[i, j])
                X3 = delta[j] - A3 * D3

                chaos = (
                    4
                    * torch.rand(1, device=self.device).item()
                    * (1 - torch.rand(1, device=self.device).item())
                )
                self.positions[i, j] = (X1 + X2 + X3) / 3 + chaos * 0.1

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Memetic(SwarmOptimizer):
    """Memetic Algorithm (Memetic) optimizer for PyTorch models."""

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
            raise ValueError("Memetic requires a closure function")
        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            fitness[i] = closure().detach()
        return fitness

    def _local_search(self, position: torch.Tensor, closure: Any) -> torch.Tensor:
        best_pos = position.clone()
        self._set_params(best_pos)
        best_fit = closure().detach()

        for _ in range(5):
            new_pos = best_pos + torch.randn_like(best_pos) * 0.01
            self._set_params(new_pos)
            new_fit = closure().detach()
            if new_fit < best_fit:
                best_fit = new_fit
                best_pos = new_pos.clone()

        return best_pos

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
            parent_idx = torch.randint(0, self.swarm_size, (1,)).item()
            child = (self.positions[i] + self.positions[parent_idx]) / 2
            self.positions[i] = self._local_search(child, closure)

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)


class Clonalg(SwarmOptimizer):
    """Clonal Selection Algorithm (Clonalg) optimizer for PyTorch models."""

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
            raise ValueError("Clonalg requires a closure function")
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

        n_select = self.swarm_size // 3
        selected = self.positions[sorted_idx[:n_select]]

        for i in range(n_select):
            clone = selected[i].clone() + torch.randn_like(selected[i]) * 0.1
            self._set_params(clone)
            clone_fit = closure().detach()

            if clone_fit < fitness[sorted_idx[i]]:
                self.positions[sorted_idx[i]] = clone
                fitness[sorted_idx[i]] = clone_fit

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        return super().step(closure)

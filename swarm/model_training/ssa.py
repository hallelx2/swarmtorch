from typing import Any

import torch

from swarmtorch.base import SwarmOptimizer


class SSA(SwarmOptimizer):
    """Sparrow Search Algorithm (SSA) optimizer for PyTorch models.

    SSA simulates the foraging and anti-predator behavior of sparrows.
    It uses producers (leaders) and scroungers (followers) with
    discovery and joining behaviors.

    Args:
        params: Model parameters to optimize.
        swarm_size: Number of sparrows in the swarm (default: 30).
        device: Device to run computations on (default: "cpu").

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = SSA(model.parameters(), swarm_size=30)
        >>> for data, target in dataloader:
        ...     def closure():
        ...         output = model(data)
        ...         return loss_fn(output, target)
        ...     optimizer.zero_grad()
        ...     optimizer.step(closure)
    """

    def __init__(
        self,
        params: Any,
        swarm_size: int = 30,
        device: str = "cpu",
    ) -> None:
        dict(
            swarm_size=swarm_size,
            device=device,
        )
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        """Initialize sparrow positions and fitness."""
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]

        self.positions = torch.rand(
            self.swarm_size,
            param_shape[0],
            device=self.device,
        )

        self.fitness = torch.full(
            (self.swarm_size,),
            float("inf"),
            device=self.device,
        )

        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)

    def _set_params(self, flat_params: torch.Tensor) -> None:
        """Set model parameters from flattened tensor."""
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.data.copy_(flat_params[idx : idx + numel].reshape(p.shape))
                idx += numel

    def _get_params(self) -> torch.Tensor:
        """Get flattened model parameters."""
        params = []
        for group in self.param_groups:
            for p in group["params"]:
                params.append(p.data.flatten())
        return torch.cat(params)

    def _evaluate_fitness(
        self,
        particles: torch.Tensor,
        closure: Any | None = None,
    ) -> torch.Tensor:
        """Evaluate fitness for each sparrow using the closure."""
        if closure is None:
            raise ValueError("SSA requires a closure function to evaluate fitness")

        fitness = torch.zeros(particles.shape[0], device=self.device)

        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            loss = closure()
            fitness[i] = loss.detach()

        return fitness

    def _update_positions(self) -> None:
        """Update sparrow positions based on SSA equations."""
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.positions, closure)

        sorted_indices = torch.argsort(fitness)
        worst_indices = sorted_indices[-int(self.swarm_size * 0.2) :]
        best_indices = sorted_indices[: int(self.swarm_size * 0.1)]

        best_idx = sorted_indices[0]
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()

        max_iter = 1000
        pd = 0.8

        for i in range(self.swarm_size):
            if i in best_indices:
                r1 = torch.rand(1, device=self.device).item()
                if r1 < pd:
                    self.positions[i] = (
                        self.positions[i]
                        * torch.exp(
                            -self.iteration_count
                            / (torch.rand(1, device=self.device) * max_iter)
                        ).item()
                        + torch.randn(self.positions.shape[1], device=self.device)
                        * 0.01
                    )
                else:
                    self.positions[i] = self.best_position + torch.randn(
                        self.positions.shape[1], device=self.device
                    ) * torch.rand(1, device=self.device)
            elif i in worst_indices:
                self.positions[i] = (
                    self.best_position.reshape(1, -1)
                    + torch.randn(self.positions.shape[1], device=self.device)
                    * torch.abs(self.positions[i] - self.best_position)
                    * (torch.rand(1, device=self.device).item() - 0.5)
                    * 2
                ).squeeze()
            else:
                self.positions[i] = self.positions[i] + (
                    torch.rand(1, device=self.device).item() - 0.5
                ) * 2 * (self.positions[i] - self.best_position) / (
                    fitness[i] - self.best_fitness + 1e-10
                )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any | None = None) -> Any:
        """Perform one SSA optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self._current_closure = closure
        return super().step(closure)

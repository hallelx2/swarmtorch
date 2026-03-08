from typing import Any

import torch

from swarmtorch.base import SwarmOptimizer


class GWO(SwarmOptimizer):
    """Grey Wolf Optimizer (GWO) optimizer for PyTorch models.

    GWO is a swarm intelligence algorithm that simulates the hunting behavior
    and social hierarchy of grey wolves. The algorithm maintains three best
    solutions (alpha, beta, delta) that guide the search, with omega wolves
    following their leaders.

    Args:
        params: Model parameters to optimize.
        swarm_size: Number of wolves in the pack (default: 30).
        device: Device to run computations on (default: "cpu").

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = GWO(model.parameters(), swarm_size=30)
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
        """Initialize wolf positions and fitness."""
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

        self.alpha_position = torch.zeros(param_shape[0], device=self.device)
        self.alpha_fitness = torch.tensor(float("inf"), device=self.device)

        self.beta_position = torch.zeros(param_shape[0], device=self.device)
        self.delta_position = torch.zeros(param_shape[0], device=self.device)

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
        """Evaluate fitness for each wolf using the closure."""
        if closure is None:
            raise ValueError("GWO requires a closure function to evaluate fitness")

        fitness = torch.zeros(particles.shape[0], device=self.device)

        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            loss = closure()
            fitness[i] = loss.detach()

        return fitness

    def _update_positions(self) -> None:
        """Update wolf positions based on alpha, beta, and delta."""
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.positions, closure)

        sorted_indices = torch.argsort(fitness)

        if fitness[sorted_indices[0]] < self.alpha_fitness:
            self.alpha_fitness = fitness[sorted_indices[0]]
            self.alpha_position = self.positions[sorted_indices[0]].clone()

        if fitness[sorted_indices[1]] < self.alpha_fitness:
            self.beta_position = self.positions[sorted_indices[1]].clone()

        if fitness[sorted_indices[2]] < self.alpha_fitness:
            self.delta_position = self.positions[sorted_indices[2]].clone()

        a = 2 - self.iteration_count * (2 / 1000)

        for i in range(self.swarm_size):
            for j in range(self.positions.shape[1]):
                r1 = torch.rand(1, device=self.device).item()
                r2 = torch.rand(1, device=self.device).item()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D1 = torch.abs(C1 * self.alpha_position[j] - self.positions[i, j])
                X1 = self.alpha_position[j] - A1 * D1

                r1 = torch.rand(1, device=self.device).item()
                r2 = torch.rand(1, device=self.device).item()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D2 = torch.abs(C2 * self.beta_position[j] - self.positions[i, j])
                X2 = self.beta_position[j] - A2 * D2

                r1 = torch.rand(1, device=self.device).item()
                r2 = torch.rand(1, device=self.device).item()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D3 = torch.abs(C3 * self.delta_position[j] - self.positions[i, j])
                X3 = self.delta_position[j] - A3 * D3

                self.positions[i, j] = (X1 + X2 + X3) / 3

        self._set_params(self.alpha_position)
        self.iteration_count += 1

    def step(self, closure: Any | None = None) -> Any:
        """Perform one GWO optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self._current_closure = closure
        return super().step(closure)

from typing import Any

import torch

from swarmtorch.base import SwarmOptimizer


class WOA(SwarmOptimizer):
    """Whale Optimization Algorithm (WOA) optimizer for PyTorch models.

    WOA simulates the hunting behavior of humpback whales. The algorithm
    uses three phases: encircling prey, bubble-net attacking, and searching
    for prey.

    Args:
        params: Model parameters to optimize.
        swarm_size: Number of whales in the swarm (default: 30).
        device: Device to run computations on (default: "cpu").

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = WOA(model.parameters(), swarm_size=30)
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
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.iteration_count = 0

    def _init_swarm(self) -> None:
        """Initialize whale positions and fitness."""
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

    def _update_positions(self) -> None:
        """Update whale positions based on WOA equations."""
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.positions, closure)

        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()

        a = 2 - self.iteration_count * (2 / 1000)

        for i in range(self.swarm_size):
            r1 = torch.rand(1, device=self.device).item()
            r2 = torch.rand(1, device=self.device).item()
            a_val = 2 * a * r1 - a
            c_val = 2 * r2

            p = torch.rand(1, device=self.device).item()

            if p < 0.5:
                if abs(a_val) < 1:
                    d = torch.abs(c_val * self.best_position - self.positions[i])
                    self.positions[i] = self.best_position - a_val * d
                else:
                    rand_idx = torch.randint(0, self.swarm_size, (1,)).item()
                    d = torch.abs(c_val * self.positions[rand_idx] - self.positions[i])
                    self.positions[i] = self.positions[rand_idx] - a_val * d
            else:
                l_val = torch.rand(1, device=self.device).item() * 2 - 1
                d = torch.abs(self.best_position - self.positions[i])
                self.positions[i] = (
                    d
                    * torch.exp(l_val * torch.tensor(torch.pi))
                    * torch.cos(2 * torch.tensor(torch.pi) * l_val)
                    + self.best_position
                )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any | None = None) -> Any:
        """Perform one WOA optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self._current_closure = closure
        return super().step(closure)

from typing import Any

import torch

from swarmtorch.base import SwarmOptimizer


class HHO(SwarmOptimizer):
    """Harris Hawks Optimization (HHO) optimizer for PyTorch models.

    HHO simulates the hunting behavior of Harris' hawks. The algorithm uses
    phases of exploration (perching) and exploitation (diving) with
    random strategies based on prey energy.

    Args:
        params: Model parameters to optimize.
        swarm_size: Number of hawks in the swarm (default: 30).
        device: Device to run computations on (default: "cpu").

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = HHO(model.parameters(), swarm_size=30)
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
        """Initialize hawk positions and fitness."""
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
        """Update hawk positions based on HHO equations."""
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.positions, closure)

        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()

        max_iter = 1000
        e = 2 * (1 - self.iteration_count / max_iter)

        for i in range(self.swarm_size):
            if e >= 1:
                q = torch.rand(1, device=self.device).item()
                if q >= 0.5:
                    random_hawk = torch.randint(0, self.swarm_size, (1,)).item()
                    self.positions[i] = self.positions[random_hawk] + torch.rand(
                        self.positions.shape[1], device=self.device
                    ) * 2 * (
                        torch.rand(self.positions.shape[1], device=self.device) - 0.5
                    )
                else:
                    self.positions[i] = self.best_position + torch.rand(
                        self.positions.shape[1], device=self.device
                    ) * 2 * (
                        torch.rand(self.positions.shape[1], device=self.device) - 0.5
                    )
            else:
                if e >= 0.5:
                    jump_strength = 2 * (1 - self.iteration_count / max_iter)
                    rabbit_mean = torch.mean(self.positions, dim=0)
                    self.positions[i] = (
                        self.best_position
                        - self.positions[i]
                        + (
                            rabbit_mean
                            + jump_strength
                            * 2
                            * (
                                torch.rand(self.positions.shape[1], device=self.device)
                                - 0.5
                            )
                        )
                        * (
                            torch.rand(self.positions.shape[1], device=self.device)
                            - 0.5
                        )
                    )
                else:
                    rabbit_mean = torch.mean(self.positions, dim=0)
                    self.positions[i] = self.best_position - e * torch.abs(
                        rabbit_mean
                        - 2
                        * torch.rand(self.positions.shape[1], device=self.device)
                        * self.best_position
                    )

        self._set_params(self.best_position)
        self.iteration_count += 1

    def step(self, closure: Any | None = None) -> Any:
        """Perform one HHO optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self._current_closure = closure
        return super().step(closure)

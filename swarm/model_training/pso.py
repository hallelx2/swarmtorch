from typing import Any

import torch

from swarmtorch.base import SwarmOptimizer


class PSO(SwarmOptimizer):
    """Particle Swarm Optimization (PSO) optimizer for PyTorch models.

    PSO is a swarm intelligence algorithm that simulates the social behavior
    of a flock of birds. Each particle represents a candidate solution and
    moves through the search space based on its own experience and the
    collective knowledge of the swarm.

    Args:
        params: Model parameters to optimize.
        swarm_size: Number of particles in the swarm (default: 30).
        w: Inertia weight (default: 0.7). Controls momentum.
        c1: Cognitive coefficient (default: 1.5). Personal best attraction.
        c2: Social coefficient (default: 1.5). Global best attraction.
        device: Device to run computations on (default: "cpu").

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = PSO(model.parameters(), swarm_size=30)
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
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        device: str = "cpu",
    ) -> None:
        defaults = dict(
            swarm_size=swarm_size,
            w=w,
            c1=c1,
            c2=c2,
            device=device,
        )
        super().__init__(params, swarm_size=swarm_size, device=device, w=w, c1=c1, c2=c2)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _init_swarm(self) -> None:
        """Initialize particle positions, velocities, personal bests, and global best."""
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]

        self.positions = torch.rand(
            self.swarm_size,
            param_shape[0],
            device=self.device,
        )

        self.velocities = torch.zeros_like(self.positions)

        self.personal_best_positions = self.positions.clone()
        self.personal_best_fitness = torch.full(
            (self.swarm_size,),
            float("inf"),
            device=self.device,
        )

        self.global_best_position = torch.zeros(param_shape[0], device=self.device)
        self.global_best_fitness = torch.tensor(float("inf"), device=self.device)

    def _evaluate_fitness(
        self,
        particles: torch.Tensor,
        closure: Any | None = None,
    ) -> torch.Tensor:
        """Evaluate fitness for each particle using the closure (forward pass)."""
        if closure is None:
            raise ValueError("PSO requires a closure function to evaluate fitness")

        fitness = torch.zeros(particles.shape[0], device=self.device)

        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            loss = closure()
            fitness[i] = loss.detach()

        return fitness

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

    def _update_positions(self) -> None:
        """Update particle positions and velocities based on PSO equations."""
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.positions, closure)

        improved = fitness < self.personal_best_fitness
        self.personal_best_fitness[improved] = fitness[improved]
        self.personal_best_positions[improved] = self.positions[improved]

        best_idx = torch.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].clone()

        r1 = torch.rand_like(self.positions)
        r2 = torch.rand_like(self.positions)

        self.velocities = (
            self.w * self.velocities
            + self.c1 * r1 * (self.personal_best_positions - self.positions)
            + self.c2 * r2 * (self.global_best_position - self.positions)
        )

        self.positions = self.positions + self.velocities

        best_idx = torch.argmin(self.personal_best_fitness)
        self._set_params(self.personal_best_positions[best_idx])

    def step(self, closure: Any | None = None) -> Any:
        """Perform one PSO optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self._current_closure = closure
        return super().step(closure)

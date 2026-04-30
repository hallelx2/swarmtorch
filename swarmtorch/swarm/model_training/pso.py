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
        position_clip: float | None = 5.0,
        velocity_clip: float | None = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            params, swarm_size=swarm_size, device=device, w=w, c1=c1, c2=c2, **kwargs
        )
        self.w = w
        self.c1 = c1
        self.c2 = c2
        # Bounds are expressed as multiples of the initial weight std so they
        # auto-adapt to whatever scale the user's model lives at. ``None``
        # disables clipping.
        self.position_clip = position_clip
        self.velocity_clip = velocity_clip

    def _init_swarm(self) -> None:
        """Initialize particle positions, velocities, personal bests, and global best."""
        param_shape = self._get_param_shape()
        self.swarm_size = self.defaults["swarm_size"]

        self.positions = self._init_positions(param_shape[0])

        self.velocities = torch.zeros_like(self.positions)

        # Reference scale derived from the initial spread of the swarm —
        # used to size velocity and position clamps.
        self._init_scale = self.positions.std().clamp(min=1e-3).item()

        self.personal_best_positions = self.positions.clone()
        self.personal_best_fitness = torch.full(
            (self.swarm_size,),
            float("inf"),
            device=self.device,
        )

        self.global_best_position = torch.zeros(param_shape[0], device=self.device)
        self.global_best_fitness = torch.tensor(float("inf"), device=self.device)

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

        if self.velocity_clip is not None:
            v_max = self.velocity_clip * self._init_scale
            self.velocities.clamp_(-v_max, v_max)

        self.positions = self.positions + self.velocities

        if self.position_clip is not None:
            x_max = self.position_clip * self._init_scale
            self.positions.clamp_(-x_max, x_max)

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

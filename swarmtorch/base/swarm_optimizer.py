from typing import Any

import torch
from torch.optim import Optimizer


class SwarmOptimizer(Optimizer):
    """Base class for swarm-based optimization algorithms.

    Inherits from torch.optim.Optimizer to leverage parameter group management,
    state dict, and compatibility with PyTorch ecosystem.

    Note: These optimizers are gradient-free - they use loss values as fitness
    rather than gradients to update weights.
    """

    uses_gradients: bool = False

    def __init__(
        self,
        params: Any,
        swarm_size: int = 30,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        defaults = dict(
            swarm_size=swarm_size,
            device=device,
            **kwargs,
        )
        super().__init__(params, defaults)
        self._initialized = False
        self._swarm_initialized = False
        self.device = torch.device(device)

    def _get_param_shape(self) -> torch.Size:
        """Get the shape of flattened parameters."""
        total_params = 0
        for group in self.param_groups:
            for p in group["params"]:
                total_params += p.numel()
        return torch.Size([total_params])

    def _set_params(self, flat_params: torch.Tensor) -> None:
        """Set model parameters from a flattened tensor.

        Args:
            flat_params: 1-D tensor containing all parameter values.
        """
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                numel = p.numel()
                p.data.copy_(flat_params[idx : idx + numel].reshape(p.shape))
                idx += numel

    def _get_params(self) -> torch.Tensor:
        """Get all model parameters as a single flattened tensor.

        Returns:
            1-D tensor of concatenated, flattened parameters.
        """
        return torch.cat(
            [p.data.flatten() for group in self.param_groups for p in group["params"]]
        )

    def _evaluate_fitness(
        self,
        particles: torch.Tensor,
        closure: Any = None,
    ) -> torch.Tensor:
        """Evaluate fitness for each particle/individual using the closure.

        Args:
            particles: 2-D tensor of shape ``(n, d)`` where each row is a
                candidate solution (flattened model parameters).
            closure: A callable that evaluates the model and returns the loss.

        Returns:
            1-D tensor of fitness (loss) values, one per particle.

        Raises:
            ValueError: If *closure* is ``None``.
        """
        if closure is None:
            raise ValueError(
                f"{type(self).__name__} requires a closure function to evaluate fitness"
            )
        fitness = torch.zeros(particles.shape[0], device=self.device)
        for i in range(particles.shape[0]):
            self._set_params(particles[i])
            fitness[i] = closure().detach()
        return fitness

    def _init_swarm(self) -> None:
        """Initialize swarm particles. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _init_swarm")

    def _update_positions(self) -> None:
        """Update particle positions. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _update_positions")

    def step(self, closure: Any = None) -> Any:
        """Perform one optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Required for swarm optimizers since they need loss values.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        if not self._swarm_initialized:
            self._init_swarm()
            self._swarm_initialized = True

        loss: Any = None
        if closure is not None:
            loss = closure()

        self._update_positions()

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients for all parameter groups."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def state_dict(self) -> dict:
        """Return the state of the optimizer as a dict."""
        state = super().state_dict()
        state["_swarm_initialized"] = self._swarm_initialized
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the optimizer state."""
        self._swarm_initialized = state_dict.pop("_swarm_initialized", False)
        super().load_state_dict(state_dict)

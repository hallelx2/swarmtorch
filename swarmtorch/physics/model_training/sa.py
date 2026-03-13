from typing import Any
import torch
from swarmtorch.base import SwarmOptimizer


class SA(SwarmOptimizer):
    """Simulated Annealing (SA) optimizer for PyTorch models.

    SA is a probabilistic optimization algorithm inspired by the annealing
    process in metallurgy.
    """

    def __init__(
        self,
        params: Any,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.95,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            params,
            swarm_size=1,
            device=device,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
        )
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.temperature = initial_temp
        self.current_params: torch.Tensor | None = None
        self.current_loss: float | None = None

    def _init_swarm(self) -> None:
        self.current_params = self._get_params().clone()
        self.temperature = self.initial_temp

    def _update_positions(self) -> None:
        closure = getattr(self, "_current_closure", None)
        if closure is None or self.current_params is None:
            return

        self._set_params(self.current_params)
        current_loss_val = closure().item()

        new_params = (
            self.current_params
            + torch.randn_like(self.current_params)
            * self.temperature
            / self.initial_temp
        )
        self._set_params(new_params)
        new_loss_val = closure().item()

        delta = new_loss_val - current_loss_val

        if (
            delta < 0
            or torch.rand(1).item()
            < torch.exp(torch.tensor(-delta / self.temperature)).item()
        ):
            self.current_params = new_params.clone()
            self.current_loss = new_loss_val
        else:
            self._set_params(self.current_params)

        self.temperature *= self.cooling_rate

    def step(self, closure: Any = None) -> Any:
        self._current_closure = closure
        if not self._swarm_initialized:
            self._init_swarm()
            self._swarm_initialized = True

        loss = None
        if closure is not None:
            loss = closure()

        self._update_positions()

        return loss

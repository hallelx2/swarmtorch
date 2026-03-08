from typing import Any

import torch

from swarmtorch.base import SwarmOptimizer


class DE(SwarmOptimizer):
    """Differential Evolution (DE) optimizer for PyTorch models.

    DE is an evolutionary algorithm that optimizes a problem by iteratively
    trying to improve a candidate solution with regard to a given measure
    of quality. It uses vector differences to perturb the population.

    Args:
        params: Model parameters to optimize.
        population_size: Number of individuals in the population (default: 30).
        cr: Crossover rate (default: 0.9). Probability of crossover.
        f: Mutation factor (default: 0.8). Controls amplification of difference vectors.
        device: Device to run computations on (default: "cpu").

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = DE(model.parameters(), population_size=30)
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
        population_size: int = 30,
        cr: float = 0.9,
        f: float = 0.8,
        device: str = "cpu",
    ) -> None:
        defaults = dict(
            population_size=population_size,
            cr=cr,
            f=f,
            device=device,
        )
        super().__init__(
            params, swarm_size=population_size, device=device, cr=cr, f=f
        )
        self.population_size = population_size
        self.cr = cr
        self.f = f

    def _init_swarm(self) -> None:
        """Initialize population and fitness."""
        param_shape = self._get_param_shape()
        self.population_size = self.defaults["swarm_size"]

        self.population = torch.rand(
            self.population_size,
            param_shape[0],
            device=self.device,
        )

        self.fitness = torch.full(
            (self.population_size,),
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
        population: torch.Tensor,
        closure: Any | None = None,
    ) -> torch.Tensor:
        """Evaluate fitness for each individual using the closure."""
        if closure is None:
            raise ValueError("DE requires a closure function to evaluate fitness")

        fitness = torch.zeros(population.shape[0], device=self.device)

        for i in range(population.shape[0]):
            self._set_params(population[i])
            loss = closure()
            fitness[i] = loss.detach()

        return fitness

    def _update_positions(self) -> None:
        """Update population using differential mutation and crossover."""
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.population, closure)
        self.fitness = fitness

        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.population[best_idx].clone()

        new_population = self.population.clone()

        for i in range(self.population_size):
            indices = torch.randperm(self.population_size)
            indices = indices[indices != i]

            if len(indices) >= 3:
                a, b, c = indices[:3]
                mutant = self.population[a] + self.f * (self.population[b] - self.population[c])

                j_rand = torch.randint(0, self.population.shape[1], (1,)).item()

                trial = torch.where(
                    torch.rand(self.population.shape[1], device=self.device) < self.cr,
                    mutant,
                    self.population[i],
                )

                self._set_params(trial)
                trial_fitness = closure().detach()

                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness

        self.population = new_population

        best_idx = torch.argmin(self.fitness)
        self._set_params(self.population[best_idx])

    def step(self, closure: Any | None = None) -> Any:
        """Perform one DE optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self._current_closure = closure
        return super().step(closure)

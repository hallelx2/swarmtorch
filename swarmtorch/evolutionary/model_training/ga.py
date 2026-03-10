from typing import Any

import torch

from swarmtorch.base import SwarmOptimizer


class GA(SwarmOptimizer):
    """Genetic Algorithm (GA) optimizer for PyTorch models.

    GA is an evolutionary algorithm that simulates natural selection.
    It uses selection, crossover, and mutation operators to evolve
    candidate solutions.

    Args:
        params: Model parameters to optimize.
        population_size: Number of individuals in population (default: 30).
        crossover_rate: Probability of crossover (default: 0.9).
        mutation_rate: Probability of mutation (default: 0.1).
        device: Device to run computations on (default: "cpu").

    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = GA(model.parameters(), population_size=30)
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
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            params,
            swarm_size=population_size,
            device=device,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
        )
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.iteration_count = 0

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

    def _selection(self) -> torch.Tensor:
        """Tournament selection."""
        selected = []
        for _ in range(self.population_size):
            indices = torch.randint(0, self.population_size, (3,))
            best_idx = indices[torch.argmin(self.fitness[indices])]
            selected.append(self.population[best_idx])
        return torch.stack(selected)

    def _crossover(self, population: torch.Tensor) -> torch.Tensor:
        """Single-point crossover."""
        new_population = []
        for i in range(0, self.population_size, 2):
            if i + 1 >= self.population_size:
                new_population.append(population[i])
                break

            if torch.rand(1, device=self.device).item() < self.crossover_rate:
                point = torch.randint(1, population.shape[1], (1,)).item()
                child1 = torch.cat([population[i][:point], population[i + 1][point:]])
                child2 = torch.cat([population[i + 1][:point], population[i][point:]])
                new_population.extend([child1, child2])
            else:
                new_population.extend([population[i], population[i + 1]])

        return torch.stack(new_population[: self.population_size])

    def _mutate(self, population: torch.Tensor) -> torch.Tensor:
        """Gaussian mutation."""
        mask = torch.rand_like(population) < self.mutation_rate
        mutation = torch.randn_like(population) * 0.1
        return torch.where(mask, population + mutation, population)

    def _update_positions(self) -> None:
        """Update population using GA operators."""
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return

        fitness = self._evaluate_fitness(self.population, closure)
        self.fitness = fitness

        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.population[best_idx].clone()

        selected = self._selection()
        crossed = self._crossover(selected)
        mutated = self._mutate(crossed)

        self.population = mutated

        best_idx = torch.argmin(self.fitness)
        self._set_params(self.population[best_idx])
        self.iteration_count += 1

    def step(self, closure: Any | None = None) -> Any:
        """Perform one GA optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self._current_closure = closure
        return super().step(closure)

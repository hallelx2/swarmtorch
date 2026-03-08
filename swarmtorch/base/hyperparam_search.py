from collections.abc import Callable
from typing import Any

import torch


class HyperparameterSearch:
    """Base class for hyperparameter search using metaheuristic algorithms.

    Provides a common interface for using swarm/evolutionary algorithms
    to search for optimal hyperparameters.
    """

    def __init__(
        self,
        model_fn: Callable[[dict], torch.nn.Module],
        param_space: dict[str, tuple[Any, ...]],
        train_fn: Callable[[torch.nn.Module, dict], float],
        iterations: int = 50,
        swarm_size: int = 30,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        """Initialize hyperparameter search.

        Args:
            model_fn: Function that creates a model given hyperparameters.
                Signature: (hyperparams: dict) -> torch.nn.Module
            param_space: Dictionary defining the search space.
                Format: {'param_name': (min, max)} for floats,
                       {'param_name': [val1, val2, ...]} for discrete
            train_fn: Function to train model and return score.
                Signature: (model: torch.nn.Module, hyperparams: dict) -> float
            iterations: Number of search iterations.
            swarm_size: Number of particles/agents in the swarm.
            device: Device to run computations on.
            verbose: Whether to print progress.
        """
        self.model_fn = model_fn
        self.param_space = param_space
        self.train_fn = train_fn
        self.iterations = iterations
        self.swarm_size = swarm_size
        self.device = torch.device(device)
        self.verbose = verbose
        self.best_params: dict | None = None
        self.best_score: float | None = None

    def _encode_params(self, params: dict) -> torch.Tensor:
        """Encode hyperparameters to a tensor."""
        encoded = []
        for key, value in self.param_space.items():
            if isinstance(value, list):
                idx = value.index(params[key]) if params[key] in value else 0
                encoded.append(idx / len(value))
            else:
                normalized = (params[key] - value[0]) / (value[1] - value[0])
                encoded.append(normalized)
        return torch.tensor(encoded, dtype=torch.float32, device=self.device)

    def _decode_params(self, encoded: torch.Tensor) -> dict:
        """Decode tensor back to hyperparameters."""
        decoded = {}
        for i, (key, value) in enumerate(self.param_space.items()):
            if isinstance(value, list):
                idx = int(encoded[i].item() * (len(value) - 1))
                decoded[key] = value[min(idx, len(value) - 1)]
            else:
                decoded[key] = value[0] + encoded[i].item() * (value[1] - value[0])
        return decoded

    def _evaluate(self, encoded: torch.Tensor) -> float:
        """Evaluate a single set of hyperparameters."""
        params = self._decode_params(encoded)
        model = self.model_fn(params)
        model = model.to(self.device)
        score = self.train_fn(model, params)
        return score

    def search(self) -> dict[str, Any]:
        """Run the hyperparameter search.

        Returns:
            Dictionary containing best hyperparameters found.
        """
        raise NotImplementedError("Subclasses must implement search")

    def _log(self, iteration: int, score: float, params: dict) -> None:
        """Log progress if verbose is enabled."""
        if self.verbose:
            print(f"Iteration {iteration}: score={score:.4f}, params={params}")

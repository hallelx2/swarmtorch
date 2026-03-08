from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from swarmtorch.base.hyperparam_search import HyperparameterSearch
from swarmtorch.base.swarm_optimizer import SwarmOptimizer


class _DummyModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Initializing close to 0.5 to keep it in the center of [0, 1] bounds initially
        self.params = nn.Parameter(torch.rand(dim))


class GenericSwarmSearch(HyperparameterSearch):
    """Generic adapter to use any SwarmOptimizer for hyperparameter tuning.
    
    This class wraps a standard SwarmOptimizer (like PSO, GWO) and allows it
    to optimize hyperparameters that are continuous, discrete, or categorical
    by providing a translation layer (a dummy parameter space in [0, 1]).
    """

    def __init__(
        self,
        optimizer_class: type[SwarmOptimizer],
        model_fn: Callable[[dict], torch.nn.Module],
        param_space: dict[str, tuple[Any, ...] | list[Any]],
        train_fn: Callable[[torch.nn.Module, dict], float],
        iterations: int = 50,
        swarm_size: int = 30,
        device: str = "cpu",
        verbose: bool = True,
        **optimizer_kwargs: Any
    ) -> None:
        super().__init__(
            model_fn=model_fn,
            param_space=param_space,
            train_fn=train_fn,
            iterations=iterations,
            swarm_size=swarm_size,
            device=device,
            verbose=verbose,
        )
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def search(self) -> dict[str, Any]:
        """Run the generic hyperparameter search using the wrapped optimizer."""
        dim = len(self.param_space)
        dummy_model = _DummyModel(dim).to(self.device)
        
        import inspect
        
        kwargs = self.optimizer_kwargs.copy()
        sig = inspect.signature(self.optimizer_class.__init__)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        
        if "device" in params or has_varkw:
            kwargs["device"] = str(self.device)
            
        if "swarm_size" in params or has_varkw:
            kwargs["swarm_size"] = self.swarm_size
        elif "population_size" in params:
            kwargs["population_size"] = self.swarm_size
            
        # Initialize the raw metaheuristic optimizer
        optimizer = self.optimizer_class(
            dummy_model.parameters(),
            **kwargs
        )
        
        # We need to capture the global best ourselves since optimizers track it differently
        global_best_encoded = torch.zeros(dim, device=self.device)
        global_best_score = float("inf")
        
        for iteration in range(self.iterations):
            def closure() -> torch.Tensor:
                nonlocal global_best_score, global_best_encoded
                
                # Enforce [0, 1] bounds directly on the dummy parameters
                with torch.no_grad():
                    dummy_model.params.clamp_(0.0, 1.0)
                    encoded = dummy_model.params.clone().detach()
                    
                try:
                    score = self._evaluate(encoded)
                except Exception:
                    score = float("inf")
                    
                # Track best score explicitly, as different optimizers might lose it or track it differently
                if score < global_best_score:
                    global_best_score = score
                    global_best_encoded = encoded.clone()
                    
                return torch.tensor(score, device=self.device, requires_grad=False)
                
            # Perform one optimization step
            optimizer.step(closure)
            
            # Help the optimizer stay within bounds by clamping its internal representations if possible
            if hasattr(optimizer, "positions"):
                with torch.no_grad():
                    optimizer.positions.clamp_(0.0, 1.0)
            elif hasattr(optimizer, "population"):
                with torch.no_grad():
                    optimizer.population.clamp_(0.0, 1.0)
            
            # Log current best
            self._log(iteration, global_best_score, self._decode_params(global_best_encoded))
            
        self.best_params = self._decode_params(global_best_encoded)
        self.best_score = global_best_score
        return self.best_params

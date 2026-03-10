# Base Classes

The foundational classes that all optimizers and searchers build upon.

## SwarmOptimizer

The base class for all model-training optimizers. Inherits from `torch.optim.Optimizer`.

```python
from swarmtorch.base import SwarmOptimizer
```

### Description

`SwarmOptimizer` provides the foundation for gradient-free optimization. All 60 model-training optimizers inherit from this class.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `swarm_size` | int | 30 | Number of particles/individuals |
| `device` | str | "cpu" | Device for computation |

### Methods

#### `_set_params(flat_params: torch.Tensor)`

Set model parameters from a flattened tensor.

#### `_get_params() -> torch.Tensor`

Get model parameters as a flattened tensor.

#### `_evaluate_fitness(particles: torch.Tensor, closure: Callable) -> torch.Tensor`

Evaluate fitness for each particle.

#### `_init_swarm()`

Initialize the swarm. Override in subclasses.

#### `_update_positions()`

Update particle positions. Override in subclasses.

### Example

```python
import torch.nn as nn
from swarmtorch.base import SwarmOptimizer

class MyOptimizer(SwarmOptimizer):
    def __init__(self, params, swarm_size=30, device="cpu"):
        super().__init__(params, swarm_size=swarm_size, device=device)
    
    def _init_swarm(self):
        # Initialize swarm
        pass
    
    def _update_positions(self):
        # Update positions
        pass
```

---

## HyperparameterSearch

Abstract base class for hyperparameter optimization.

```python
from swarmtorch.base import HyperparameterSearch
```

### Description

Base class for HPO. Provides encoding/decoding of parameter spaces and evaluation framework.

### Methods

#### `search() -> dict`

Run hyperparameter search. Must be implemented by subclasses.

#### `_encode_params(encoded: torch.Tensor) -> dict`

Encode normalized [0,1] parameters to actual hyperparameter values.

#### `_decode_params(params: dict) -> torch.Tensor`

Decode hyperparameter values to normalized [0,1] tensor.

---

## GenericSwarmSearch

Adapter that wraps any `SwarmOptimizer` for hyperparameter tuning.

```python
from swarmtorch.base import GenericSwarmSearch
```

### Description

The key innovation: wraps model-training optimizers to work as HPO searchers by creating a dummy model in [0,1] parameter space.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_cls` | type | Required | The optimizer class to wrap |
| `model_fn` | callable | Required | Function to build model from params |
| `param_space` | dict | Required | Hyperparameter search space |
| `train_fn` | callable | Required | Training function returning loss |
| `iterations` | int | 50 | Number of search iterations |
| `swarm_size` | int | 10 | Candidates per iteration |
| `device` | str | "cpu" | Device for computation |
| `verbose` | bool | True | Print progress |

### Parameter Space Format

```python
param_space = {
    'lr': (0.001, 0.1),           # Continuous: (min, max)
    'hidden_dim': [16, 32, 64],    # Categorical: list of choices
}
```

### Example

```python
from swarmtorch.base import GenericSwarmSearch
from swarmtorch import PSO

class PSOSearch(GenericSwarmSearch):
    def __init__(self, *args, **kwargs):
        super().__init__(PSO, *args, **kwargs)

searcher = PSOSearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1)},
    train_fn=train_fn,
    iterations=20,
)
best = searcher.search()
```

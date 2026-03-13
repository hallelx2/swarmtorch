# Creating Custom Optimizers

This guide shows how to create your own swarm optimizer by extending `SwarmOptimizer`.

## Basic Structure

Every optimizer must inherit from `SwarmOptimizer` and implement these methods:

```python
from swarmtorch.base import SwarmOptimizer

class MyOptimizer(SwarmOptimizer):
    
    def __init__(self, params, swarm_size=30, device="cpu", **kwargs):
        super().__init__(params, swarm_size=swarm_size, device=device, **kwargs)
        # Your initialization
    
    def _init_swarm(self):
        # Initialize swarm/population
        # Example: positions, velocities, best positions, etc.
        pass
    
    def _update_positions(self):
        # Update particle positions based on your algorithm
        # This is called each optimization step
        pass
```

## Step-by-Step Example: Simplified PSO

Let's implement a minimal PSO as a learning example:

```python
import torch
from swarmtorch.base import SwarmOptimizer

class SimplePSO(SwarmOptimizer):
    """Simplified Particle Swarm Optimization."""
    
    def __init__(self, params, swarm_size=30, w=0.7, c1=1.5, c2=1.5, device="cpu"):
        super().__init__(params, swarm_size=swarm_size, device=device)
        self.w = w      # Inertia weight
        self.c1 = c1    # Cognitive coefficient
        self.c2 = c2    # Social coefficient
    
    def _init_swarm(self):
        """Initialize particle positions and velocities."""
        param_shape = self._get_param_shape()
        
        # Random positions in [0, 1]
        self.positions = torch.rand(
            self.swarm_size, param_shape[0], device=self.device
        )
        
        # Zero velocities
        self.velocities = torch.zeros_like(self.positions)
        
        # Track personal bests
        self.personal_best_positions = self.positions.clone()
        self.personal_best_fitness = torch.full(
            (self.swarm_size,), float("inf"), device=self.device
        )
        
        # Global best
        self.global_best_position = torch.zeros(param_shape[0], device=self.device)
        self.global_best_fitness = torch.tensor(float("inf"), device=self.device)
    
    def _update_positions(self):
        """Update positions using PSO equations."""
        # Get current closure for evaluation
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return
        
        # Evaluate fitness for all particles
        fitness = self._evaluate_fitness(self.positions, closure)
        
        # Update personal bests
        improved = fitness < self.personal_best_fitness
        self.personal_best_fitness[improved] = fitness[improved]
        self.personal_best_positions[improved] = self.positions[improved]
        
        # Update global best
        best_idx = torch.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].clone()
        
        # Update velocities
        r1 = torch.rand_like(self.positions)
        r2 = torch.rand_like(self.positions)
        
        self.velocities = (
            self.w * self.velocities
            + self.c1 * r1 * (self.personal_best_positions - self.positions)
            + self.c2 * r2 * (self.global_best_position - self.positions)
        )
        
        # Update positions
        self.positions = self.positions + self.velocities
        
        # Apply best position to model
        best_idx = torch.argmin(self.personal_best_fitness)
        self._set_params(self.personal_best_positions[best_idx])
```

## Required Methods

### `__init__(self, params, ...)`
Initialize optimizer parameters. Call `super().__init__()` with:
- `params`: Model parameters
- `swarm_size`: Number of particles
- `device`: Computation device

### `_init_swarm(self)`
Initialize swarm/population. Set up:
- `self.positions`: Tensor of shape `(swarm_size, num_params)`
- Best position tracking (per-particle and global)
- Any algorithm-specific state

### `_update_positions(self)`
Update particle positions. Steps:
1. Get current closure via `getattr(self, "_current_closure", None)`
2. Evaluate fitness using `self._evaluate_fitness(positions, closure)`
3. Update best positions
4. Update particle positions
5. Apply best to model using `self._set_params()`

## Helper Methods (Inherited)

The base class provides these helpers:

```python
# Get flattened parameter shape
shape = self._get_param_shape()

# Set model parameters from flattened tensor
self._set_params(flat_params)

# Get model parameters as flattened tensor
params = self._get_params()

# Evaluate fitness for all particles (requires closure)
fitness = self._evaluate_fitness(particles, closure)
```

## Complete Example: Custom "Random Search"

```python
import torch
from swarmtorch.base import SwarmOptimizer

class RandomSearchOptimizer(SwarmOptimizer):
    """Random search optimizer - each step picks random new position."""
    
    def __init__(self, params, swarm_size=30, device="cpu"):
        super().__init__(params, swarm_size=swarm_size, device=device)
    
    def _init_swarm(self):
        param_shape = self._get_param_shape()
        self.positions = torch.rand(
            self.swarm_size, param_shape[0], device=self.device
        )
        self.best_position = torch.zeros(param_shape[0], device=self.device)
        self.best_fitness = torch.tensor(float("inf"), device=self.device)
    
    def _update_positions(self):
        closure = getattr(self, "_current_closure", None)
        if closure is None:
            return
        
        fitness = self._evaluate_fitness(self.positions, closure)
        
        best_idx = torch.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_position = self.positions[best_idx].clone()
        
        # Random search: explore new positions
        self.positions = torch.rand(
            self.swarm_size, self.positions.shape[1], device=self.device
        )
        
        self._set_params(self.best_position)
```

## Using Your Custom Optimizer

```python
import torch.nn as nn

model = nn.Linear(10, 2)
optimizer = SimplePSO(model.parameters(), swarm_size=30)

def closure():
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    return loss

for _ in range(100):
    optimizer.step(closure)
```

## Creating Custom HPO Searcher

Any optimizer can be used for HPO via `GenericSwarmSearch`:

```python
from swarmtorch.base import GenericSwarmSearch

class MyOptimizerSearch(GenericSwarmSearch):
    """HPO using my custom optimizer."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(MyOptimizer, *args, **kwargs)
```

## Best Practices

1. **Inherit from `SwarmOptimizer`**: Don't reinvent the wheel
2. **Use inherited methods**: `_set_params`, `_get_params`, `_evaluate_fitness`
3. **Implement `_init_swarm` and `_update_positions`**: These are required
4. **Handle None closure**: Always check if closure exists
5. **Track best positions**: Update and apply the best found solution

## Testing Your Optimizer

```python
# Quick test
model = nn.Linear(4, 1)
opt = MyOptimizer(model.parameters(), swarm_size=10)

def closure():
    opt.zero_grad()
    loss = torch.tensor([0.5])
    return loss

loss = opt.step(closure)
print(f"Loss: {loss}")
```

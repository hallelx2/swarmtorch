# Concepts

This guide explains the core concepts behind SwarmTorch and how metaheuristic optimization works.

## What are Metaheuristics?

Metaheuristics are high-level problem-independent algorithms that guide other heuristics to find good solutions to optimization problems. Unlike gradient-based methods, they don't require differentiable objective functions.

### Key Characteristics

1. **Gradient-Free**: Only require the objective function value, not derivatives
2. **Population-Based**: Maintain multiple candidate solutions simultaneously
3. **Stochastic**: Use randomness to escape local optima
4. **Nature-Inspired**: Many algorithms mimic biological/physical phenomena

## Swarm Intelligence Algorithms

These algorithms simulate collective behavior of agents in nature.

### Particle Swarm Optimization (PSO)

Inspired by flocking birds. Each "particle" has:
- **Position**: Current solution
- **Velocity**: Direction and speed of movement
- **Personal Best**: Best position found by this particle
- **Global Best**: Best position found by any particle

```
v_new = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
x_new = x + v_new
```

Where:
- `w` = inertia weight
- `c1` = cognitive coefficient (personal experience)
- `c2` = social coefficient (swarm knowledge)
- `r1, r2` = random numbers [0, 1]

### Grey Wolf Optimizer (GWO)

Mimics the hunting hierarchy of grey wolves:
- **Alpha (α)**: Best solution (leader)
- **Beta (β)**: Second best (advisor)
- **Delta (δ)**: Third best (sub-leader)
- **Omega (ω)**: Remaining wolves (followers)

Wolves update positions based on alpha, beta, and delta guidance.

### Whale Optimization Algorithm (WOA)

Simulates humpback whale hunting:
- **Encircling Prey**: Approach and surround the target
- **Bubble-Net Attack**: Spiral movement toward prey
- **Search for Prey**: Random search when prey isn't found

## Evolutionary Algorithms

Inspired by biological evolution.

### Differential Evolution (DE)

1. **Mutation**: Create mutant vector: `v = x_r1 + F * (x_r2 - x_r3)`
2. **Crossover**: Mix mutant and target vectors
3. **Selection**: Keep better of trial vs target

### Genetic Algorithm (GA)

1. **Selection**: Choose parent solutions (tournament, roulette)
2. **Crossover**: Combine parents to create offspring
3. **Mutation**: Randomly modify offspring
4. **Elitism**: Keep best solutions

## Architecture

SwarmTorch provides two main interfaces:

### 1. Model Training (`SwarmOptimizer`)

```python
from swarmtorch import PSO
from swarmtorch.base import SwarmOptimizer

# All optimizers inherit from torch.optim.Optimizer
optimizer = PSO(model.parameters(), swarm_size=30)

# The closure pattern is required
def closure():
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    return loss

optimizer.step(closure)
```

### 2. Hyperparameter Tuning (`GenericSwarmSearch`)

```python
from swarmtorch import PSOSearch

searcher = PSOSearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'hidden_dim': [32, 64]},
    train_fn=train_fn,
    iterations=50,
)
best_params = searcher.search()
```

## How GenericSwarmSearch Works

The clever trick: we "trick" the model-training optimizers into doing hyperparameter search by:

1. **Creating a Dummy Model**: Parameters live in [0,1] space
2. **Encoding Hyperparameters**: Map hyperparameters to model weights
3. **Swarm "Trains" the Dummy**: Optimizer thinks it's training weights
4. **Decoding**: Convert learned weights back to hyperparameters

```
Hyperparameter Space → [Encoder] → Weight Space → Optimizer → [Decoder] → Hyperparameters
```

This means **any** model-training optimizer can be used for HPO!

## Algorithm Categories

| Category | # | Inspiration | Examples |
|----------|---|-------------|----------|
| **Swarm Intelligence** | 32 | Animal swarms, flocks | PSO, GWO, WOA, HHO |
| **Evolutionary** | 8 | Natural selection | DE, GA, CEM |
| **Physics-Based** | 3 | Physical processes | SA, GSA, FPA |
| **Bio-Inspired** | 5 | Biological systems | ALO, BBO, MVO |
| **Human-Based** | 2 | Human behavior | TLBO, HarmonySearch |
| **Hybrid** | 10 | Combined approaches | SMA, Gorilla, JSO |

## Performance Considerations

### When to Use SwarmTorch

- Non-differentiable loss functions
- Discrete/hybrid parameter spaces
- Multi-modal optimization landscapes
- When gradients are unreliable (e.g., noisy evaluation)
- Hyperparameter optimization

### When to Use Standard Optimizers

- Large-scale differentiable problems
- Fast iteration needed
- Well-behaved convex landscapes

### Tips for Best Results

1. **Swarm Size**: 20-50 typically works well
2. **Iterations**: More iterations = better exploration
3. **Loss Function**: Use `BCEWithLogitsLoss` instead of `BCEWithSigmoid`
4. **Closure**: Always provide a closure that returns the loss
5. **Seeding**: Set seeds for reproducibility

```python
torch.manual_seed(42)
import numpy as np
np.random.seed(42)
```

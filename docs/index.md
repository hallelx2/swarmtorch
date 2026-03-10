# SwarmTorch 🐝🔥

<Callout type="info">
  SwarmTorch brings 120 metaheuristic optimization algorithms to PyTorch — 60 model-training optimizers and 60 hyperparameter-tuning searchers.
</Callout>

SwarmTorch is a high-performance, academic-grade library that enables **gradient-free neural network training** and **intelligent hyperparameter optimization** using nature-inspired metaheuristic algorithms.

## Why SwarmTorch?

Traditional deep learning relies on gradient-based optimization (Adam, SGD, etc.). SwarmTorch complements these with:

- **Gradient-Free Training**: Optimize weights where gradients don't exist or are unreliable
- **Hyperparameter Optimization**: Replace grid/random search with intelligent exploration
- **60+ Algorithms**: From PSO and GWO to DE, GA, and hybrid methods
- **PyTorch Native**: Drop-in replacement for standard optimizers

## Key Features

<CardGroup cols={2}>
  <Card title="60 Model Optimizers" icon="brain">
    Train neural networks without backpropagation using swarm intelligence, evolutionary algorithms, and more.
  </Card>
  <Card title="60 HPO Searchers" icon="magnifying-glass">
    Automatically find optimal hyperparameters using nature-inspired search.
  </Card>
  <Card title="PyTorch Compatible" icon="brand-elixir">
    Uses the standard `torch.optim.Optimizer` interface. Easy to integrate.
  </Card>
  <Card title="Research-Ready" icon="chart-line">
    Benchmarked on 120+ algorithms with publication-quality visualizations.
  </Card>
</CardGroup>

## Quick Example

```python
import torch.nn as nn
from swarmtorch import PSO

# Gradient-free model training
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = PSO(model.parameters(), swarm_size=30)

def closure():
    optimizer.zero_grad()
    output = model(x)
    loss = nn.BCEWithLogitsLoss()(output, y)
    return loss

for _ in range(100):
    optimizer.step(closure)
```

## Algorithm Categories

| Category | # Algorithms | Examples |
|----------|-------------|----------|
| **Swarm Intelligence** | 32 | PSO, GWO, WOA, HHO, SSA |
| **Evolutionary** | 8 | DE, GA, CEM, PBIL |
| **Physics-Based** | 3 | SA, GSA, FPA |
| **Bio-Inspired** | 5 | ALO, BBO, MVO |
| **Human-Based** | 2 | TLBO, HarmonySearch |
| **Hybrid** | 10 | SMA, Gorilla, JSO |

## Installation

```bash
pip install swarmtorch
```

For benchmarking dependencies:
```bash
pip install swarmtorch[benchmarks]
```

## Next Steps

<CardGroup cols={2}>
  <Card title="Quickstart" icon="rocket" href="/getting-started/quickstart">
    Get running in 5 minutes with PSO training and HPO examples.
  </Card>
  <Card title="Model Training Guide" icon="book" href="/guides/model-training">
    Learn how to use gradient-free optimization for neural networks.
  </Card>
  <Card title="Hyperparameter Tuning" icon="magnifying-glass" href="/guides/hyperparameter-tuning">
    Discover how to optimize model hyperparameters automatically.
  </Card>
  <Card title="API Reference" icon="code" href="/api-reference/base">
    Explore the full API documentation.
  </Card>
</CardGroup>

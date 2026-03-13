---
title: Home
description: SwarmTorch - 120 Metaheuristic Optimization Algorithms for PyTorch
---

# SwarmTorch 🐝🔥

<Info>
SwarmTorch brings 120 metaheuristic optimization algorithms to PyTorch — 60 model-training optimizers and 60 hyperparameter-tuning searchers.
</Info>

SwarmTorch is a high-performance, academic-grade library that enables **gradient-free neural network training** and **intelligent hyperparameter optimization** using nature-inspired metaheuristic algorithms.

## Why SwarmTorch?

Traditional deep learning relies on gradient-based optimization (Adam, SGD). SwarmTorch complements these with:

| Scenario | Gradient (Adam/SGD) | SwarmTorch |
|----------|-------------------|------------|
| Standard classification | ✅ Best choice | Overkill |
| Non-differentiable loss | ❌ Can't use | ✅ Perfect fit |
| Discrete optimization | ❌ Can't use | ✅ Perfect fit |
| Multi-modal landscapes | Gets stuck | Explores well |
| Hyperparameter tuning | Manual search | Automated |

## Key Features

<CardGroup cols={2}>
  <Card title="60 Model Optimizers" icon="brain">
    Train neural networks without backpropagation using swarm intelligence, evolutionary algorithms, and more.
  </Card>
  <Card title="60 HPO Searchers" icon="magnifying-glass">
    Automatically find optimal hyperparameters using nature-inspired search.
  </Card>
  <Card title="PyTorch Native" icon="brand-elixir">
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

| Category | # | Examples |
|----------|---|----------|
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

## Get Started

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
  <Card title="Choosing an Algorithm" icon="sparkles" href="/guides/choosing-algorithm">
    Find the right algorithm for your problem.
  </Card>
  <Card title="API Reference" icon="code" href="/api-reference/base">
    Explore the full API documentation.
  </Card>
  <Card title="Benchmarks" icon="chart-bar" href="/benchmarks/benchmarks">
    See where swarm algorithms excel vs gradient methods.
  </Card>
</CardGroup>

<div align="center">

# SwarmTorch 🐝🔥
**The ultimate metaheuristic library for PyTorch**

[![CI](https://github.com/hallelx2/swarmtorch/actions/workflows/ci.yml/badge.svg)](https://github.com/hallelx2/swarmtorch/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/swarmtorch.svg)](https://badge.fury.io/py/swarmtorch)
[![Downloads](https://pepy.tech/badge/swarmtorch)](https://pepy.tech/project/swarmtorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

SwarmTorch is a high-performance, academic-grade library that brings **120+ metaheuristic algorithms** to the PyTorch ecosystem. It enables gradient-free neural network training and state-of-the-art hyperparameter optimization (HPO) with a single, unified API.

[Installation](#-installation) • [Key Features](#-key-features) • [Benchmarks & Research](#-benchmarks--research) • [Usage](#-usage) • [Citation](#-citation)

</div>

---

## 🚀 Key Features

*   **120+ Algorithms**: Categorized into Swarm Intelligence, Evolutionary, Physics-based, Human-based, Bio-inspired, and Hybrids.
*   **Gradient-Free Training**: Optimize weights for non-differentiable or complex loss landscapes directly as a PyTorch `Optimizer`.
*   **Deep HPO Integration**: Replace Random/Grid search with intelligent, nature-inspired exploration.
*   **Research Ready**: Includes full benchmarking suites, raw experimental data, and publication-quality visualizations.
*   **Highly Optimized**: Leverages PyTorch's tensor operations for swarm-level parallelism.

---

## 📈 Benchmarks & Research

We conducted a massive-scale evaluation of **118 algorithms**. Our research shows that **94.9% of SwarmTorch searchers outperform the standard Random Search baseline** in HPO tasks.

Detailed performance analysis, convergence plots, and category reliability studies are available in the dedicated benchmarks document:

👉 **[View Full Research & Benchmarks Report](benchmarks/BENCHMARKS.md)**

---

## 📦 Installation

**Using pip:**
```bash
pip install swarmtorch
```

**Using uv (Recommended):**
```bash
uv add swarmtorch
```

---

## 💻 Usage

### Model Weight Optimization
```python
import torch.nn as nn
from swarmtorch.swarm.model_training import PSO

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = PSO(model.parameters(), swarm_size=30)

def closure():
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    return loss

optimizer.step(closure)
```

### Hyperparameter Tuning
```python
from swarmtorch.swarm.hyperparameter_tuning import PSOSearch

searcher = PSOSearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'hidden_dim': [32, 64]},
    train_fn=train_fn,
    iterations=50
)
best_params = searcher.search()
```

---

## 🤝 Acknowledgments & References
This library was developed with reference to the **[pyMetaheuristic](https://github.com/mariosv/pyMetaheuristic)** library. We are grateful for their contributions to the metaheuristic optimization community.

## 📝 Citation
```bibtex
@software{swarmtorch2026,
  author = {Halleluyah Darasimi Oludele},
  title = {SwarmTorch: A PyTorch Library for 120+ Metaheuristic Optimization Algorithms},
  year = {2026},
  url = {https://github.com/hallelx2/swarmtorch}
}
```

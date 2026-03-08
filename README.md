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

[Installation](#-installation) • [Key Features](#-key-features) • [Benchmarks](#-research--benchmarking-results) • [Usage](#-usage) • [Citation](#-citation)

</div>

---

## 🚀 Key Features

*   **120+ Algorithms**: Categorized into Swarm Intelligence, Evolutionary, Physics-based, Human-based, Bio-inspired, and Hybrids.
*   **Gradient-Free Training**: Optimize weights for non-differentiable or complex loss landscapes directly as a PyTorch `Optimizer`.
*   **Deep HPO Integration**: Replace Random/Grid search with intelligent, nature-inspired exploration.
*   **Research Ready**: Includes full benchmarking suites, raw experimental data, and publication-quality visualizations.
*   **Highly Optimized**: Leverages PyTorch's tensor operations for swarm-level parallelism.

---

## 📈 Research & Benchmarking Results

We conducted a massive-scale rigorous evaluation of **118 algorithms** to benchmark their performance across various deep learning tasks.

### 1. Model Training (Weight Optimization)
SwarmTorch enables the training of neural networks without gradients. Several metaheuristics exhibit convergence stability comparable to standard gradient-based methods.

<div align="center">
  <img src="https://raw.githubusercontent.com/hallelx2/swarmtorch/master/swarmtorch/benchmarks/training_convergence.png" width="800px">
  <p><i>Figure 1: Convergence history comparing Swarm Optimizers (PSO, HHO, etc.) against Adam and SGD.</i></p>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/hallelx2/swarmtorch/master/swarmtorch/benchmarks/top_optimizers_training.png" width="800px">
  <p><i>Figure 2: The Top 25 most effective weight optimizers ranked by final loss.</i></p>
</div>

### 2. High-Density Distribution Analysis
We analyzed the reliability of each category. Swarm and Hybrid categories demonstrated the highest stability and lowest variance across multiple trials.

<div align="center">
  <img src="https://raw.githubusercontent.com/hallelx2/swarmtorch/master/swarmtorch/benchmarks/bench_category_dist.png" width="800px">
  <p><i>Figure 3: Statistical distribution of final loss across categories. Lower loss indicates superior optimization.</i></p>
</div>

### 3. Hyperparameter Optimization (HPO) Benchmarks
Our metaheuristic searchers are designed to replace Random Search with more intelligent exploration strategies. **94.9% of our algorithms outperformed Random Search.**

<div align="center">
  <img src="https://raw.githubusercontent.com/hallelx2/swarmtorch/master/swarmtorch/benchmarks/bench_success_rate.png" width="500px">
  <p><i>Figure 4: Success rate of metaheuristic searchers vs. the Random Search baseline.</i></p>
</div>

### 4. The "Generalist" Frontier
We identified "Generalist" algorithms that excel at both weight optimization and hyperparameter tuning.

<div align="center">
  <img src="https://raw.githubusercontent.com/hallelx2/swarmtorch/master/swarmtorch/benchmarks/bench_generalist_map.png" width="800px">
  <p><i>Figure 5: Scatter plot mapping Training Efficiency vs. HPO Accuracy. Elite generalists occupy the top-right quadrant.</i></p>
</div>

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

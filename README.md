# SwarmTorch 🐝🔥
**A PyTorch Library for Metaheuristic Optimization in Deep Learning**

[![arXiv](https://img.shields.io/badge/arXiv-2503.XXXXX-red)](https://arxiv.org/abs/XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-green.svg)](https://pytorch.org/)

---

## 📄 About

SwarmTorch is a comprehensive PyTorch library that brings **120 metaheuristic optimization algorithms** to the deep learning ecosystem. It provides a unified API for:

- **60 model-training optimizers** — gradient-free weight optimization for neural networks
- **60 hyperparameter searchers** — intelligent hyperparameter optimization (HPO)

Published on arXiv: **[arXiv:2503.XXXXX](https://arxiv.org/abs/XXXXX)**

---

## 🚀 Key Features

- **120 Algorithms Across 6 Families**: Swarm Intelligence, Evolutionary, Physics-based, Human-based, Bio-inspired, and Hybrid methods
- **Native PyTorch Integration**: Drop-in replacement for `torch.optim.Optimizer`
- **GPU-Accelerated**: Full tensor parallelism using PyTorch's CUDA support
- **Research-Grade Benchmarks**: Comprehensive evaluation across 60 training algorithms and 60 HPO searchers with reproducible results

---

## 📊 Key Research Findings

Our empirical evaluation on standard benchmarks demonstrates:

- **63.3%** of SwarmTorch's hyperparameter searchers outperform Random Search baseline
- Metaheuristics achieve significantly lower loss on multimodal (non-convex) test functions where gradient-based methods fail
- Up to **17.4×** GPU speedup for models with >50K parameters

See our [paper](paper/swarmtorch_paper.pdf) for detailed results.

---

## 📦 Installation

```bash
pip install swarmtorch
```

**With development dependencies:**

```bash
pip install -e ".[dev]"
```

---

## 💻 Usage Examples

### Model Weight Optimization

```python
import torch.nn as nn
from swarmtorch import PSO

# Define model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Use PSO as optimizer
optimizer = PSO(model.parameters(), swarm_size=30)

# Training loop
for epoch in range(100):
    def closure():
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        return loss
    
    optimizer.step(closure)
```

### Hyperparameter Optimization

```python
from swarmtorch import PSOSearch

# Define search space
searcher = PSOSearch(
    model_fn=build_model,
    param_space={
        'lr': (0.001, 0.1),
        'hidden_dim': [32, 64, 128],
        'batch_size': [16, 32, 64]
    },
    train_fn=train_fn,
    iterations=50,
    swarm_size=20
)

# Run optimization
best_params = searcher.search()
```

---

## 📂 Repository Structure

```
swarmtorch/
├── swarmtorch/          # Main library
│   ├── swarm/          # Swarm Intelligence algorithms
│   ├── evolutionary/   # Evolutionary algorithms
│   ├── physics/       # Physics-based algorithms
│   ├── bio_inspired/  # Bio-inspired algorithms
│   ├── human_based/  # Human-based algorithms
│   └── hybrid/        # Hybrid algorithms
├── benchmarks/         # Experimental benchmarks
├── paper/             # Research paper & figures
└── README.md
```

---

## 📚 Benchmark Results

Detailed experimental results are available in:
- **[COMPREHENSIVE_EXPERIMENT_REPORT.md](benchmarks/COMPREHENSIVE_EXPERIMENT_REPORT.md)**
- **[BENCHMARKS.md](benchmarks/BENCHMARKS.md)**

### Top Performing HPO Searchers

| Rank | Algorithm | Best Accuracy |
|------|-----------|---------------|
| 1 | SA Search | 98.5% |
| 2 | DVBA Search | 98.5% |
| 3 | PBIL Search | 98.0% |
| 4 | AFSA Search | 98.0% |
| 5 | JSO Search | 98.0% |

### Top Performing Training Optimizers

| Rank | Algorithm | Final Loss |
|------|-----------|------------|
| 1 | Adam (baseline) | 0.033 |
| 2 | CA | 0.132 |
| 3 | HHO | 0.167 |
| 4 | CEM | 0.199 |
| 5 | PFA | 0.239 |

---

## 🔧 Hardware & Reproducibility

All benchmarks were run on:
- **GPU**: NVIDIA T4x2 (Kaggle)
- **CPU**: Intel Core i7-12700K (workstation)
- **Software**: PyTorch 2.9.0+cu126, Python 3.11, NumPy 1.26

Random seeds fixed at 42 for reproducibility.

---

## 📝 Citation

If you use SwarmTorch in your research, please cite:

```bibtex
@article{swarmtorch2026,
  author  = {Halleluyah Darasimi Oludele},
  title   = {SwarmTorch: A PyTorch Library for 120 Metaheuristic Optimization Algorithms in Deep Learning},
  journal = {arXiv preprint arXiv:2503.XXXXX},
  year    = {2026}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgments

Inspired by [pyMetaheuristic](https://github.com/mariosv/pyMetaheuristic) and the broader metaheuristic optimization community.

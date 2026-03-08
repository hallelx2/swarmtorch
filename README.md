# SwarmTorch 🐝🔥: Professional Metaheuristic Optimization for PyTorch

**SwarmTorch** is a production-grade, high-performance library designed to bridge the gap between metaheuristic optimization and deep learning. It provides a massive collection of **over 120 algorithms** (59 Model Weight Optimizers and 59 Hyperparameter Tuners) that are fully compatible with the PyTorch ecosystem.

Whether you are performing **Gradient-Free Neural Network Training** or **Automated Hyperparameter Optimization (HPO)**, SwarmTorch offers a unified interface to the world's most powerful swarm and evolutionary algorithms.

---

## 🏛️ Project Architecture & Taxonomy

The library is logically structured into 6 core categories, each reflecting a unique mathematical or biological inspiration. This organization allows researchers to easily compare different families of algorithms.

### 1. Swarm Intelligence (`swarmtorch.swarm`)

*The flagship category, modeling decentralized, self-organized collective behaviors.*

- **Top Algorithms:** PSO (Particle Swarm), GWO (Grey Wolf), HHO (Harris Hawks), SSA (Salp Swarm).
- **Best For:** Global search in high-dimensional spaces.

### 2. Evolutionary Algorithms (`swarmtorch.evolutionary`)

*Based on the mechanisms of natural selection and biological evolution.*

- **Top Algorithms:** GA (Genetic Algorithm), DE (Differential Evolution), CEM (Cross-Entropy Method).
- **Best For:** Robust, mutation-driven exploration.

### 3. Physics-Based (`swarmtorch.physics`)

*Algorithms derived from the fundamental laws of the physical world.*

- **Top Algorithms:** SA (Simulated Annealing), GSA (Gravitational Search), FPA (Flower Pollination).
- **Best For:** Problems with well-defined energy or force landscapes.

### 4. Human-Based (`swarmtorch.human_based`)

*Simulates human social interactions, teaching, and learning processes.*

- **Top Algorithms:** TLBO (Teaching-Learning-Based), Harmony Search.
- **Best For:** Knowledge-sharing-driven convergence.

### 5. Bio-Inspired (`swarmtorch.bio_inspired`)

*General biological models and life-cycle simulations.*

- **Top Algorithms:** ALO (Ant Lion), BBO (Biogeography-Based), MVO (Multi-Verse).
- **Best For:** Specialized niche optimization tasks.

### 6. Hybrid Algorithms (`swarmtorch.hybrid`)

*Advanced optimizers that combine multiple strategies for superior convergence.*

- **Top Algorithms:** SMA (Slime Mold), Gorilla Optimizer, Cat Swarm.
- **Best For:** Complex, multimodal surfaces where single-strategy algorithms might stall.

---

## 🚀 Installation & Integration

### Setup

```bash
# Clone and install as a development package
git clone https://github.com/your-repo/swarmtorch.git
cd swarmtorch
pip install -e .
```

### 1. Model Weight Optimization (Gradient-Free)

Train any PyTorch model without using `loss.backward()`. This is ideal for non-differentiable objectives or when exploring global landscapes.

```python
import torch
import torch.nn as nn
from swarmtorch.swarm.model_training import PSO

# Define your standard PyTorch model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Initialize a SwarmTorch Optimizer
# Note: uses_gradients = False
optimizer = PSO(model.parameters(), swarm_size=30)
criterion = nn.BCEWithLogitsLoss()

def closure():
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    # NO loss.backward() needed!
    return loss

# Perform an optimization step
optimizer.step(closure)
```

### 2. Hyperparameter Optimization (HPO)

Optimize architectural choices and training parameters (learning rates, layer sizes) using intelligent searchers instead of grid or random search.

```python
from swarmtorch.swarm.hyperparameter_tuning import PSOSearch

def build_model(params):
    return nn.Linear(10, int(params['hidden_dim']))

def train_fn(model, params):
    # Train and return validation loss/error to minimize
    # Higher performance = Lower return value
    return validation_error

param_space = {
    'lr': (0.001, 0.1),            # Continuous
    'hidden_dim': [32, 64, 128],   # Discrete
    'dropout': (0.0, 0.5)          # Continuous
}

searcher = PSOSearch(
    model_fn=build_model,
    param_space=param_space,
    train_fn=train_fn,
    iterations=50,
    swarm_size=20
)

best_params = searcher.search()
print(f"Optimal Configuration: {best_params}")
```

---

## 📈 Research & Benchmarking Results

Our library has undergone a massive-scale rigorous evaluation of **118 algorithms** to benchmark their performance across various deep learning tasks.

### 1. Model Training Benchmarks (Weight Optimization)
SwarmTorch enables the training of neural networks without gradients. Our tests on non-linear classification tasks show that several metaheuristics can achieve convergence comparable to standard gradient-based methods.

![Training Convergence](benchmarks/training_convergence.png)
*Figure 1: Convergence history comparing Swarm Optimizers (PSO, HHO, etc.) against Adam and SGD. Many swarm algorithms exhibit highly stable descent.*

![Top Training Optimizers](benchmarks/top_optimizers_training.png)
*Figure 2: The Top 25 most effective weight optimizers ranked by final loss. Hybrid and Swarm Intelligence algorithms show the strongest performance.*

### 2. High-Density Distribution Analysis
We analyzed the reliability of each category. Swarm and Hybrid categories demonstrated the highest stability and lowest variance across multiple trials.

![Category Distribution](benchmarks/bench_category_dist.png)
*Figure 3: Statistical distribution of final loss across categories. Lower loss and tighter boxes indicate superior and more reliable optimization.*

### 3. Hyperparameter Optimization (HPO) Benchmarks
Our metaheuristic searchers are designed to replace Random Search with more intelligent exploration strategies.

![HPO Success Rate](benchmarks/bench_success_rate.png)
*Figure 4: Success rate of metaheuristic searchers vs. the Random Search baseline. Over 94% of our algorithms outperformed Random Search.*

### 4. The "Generalist" Frontier
By mapping Training Robustness against HPO Accuracy, we identified the most versatile algorithms in the library—those that excel in both weight optimization and hyperparameter tuning.

![Generalist Mapping](benchmarks/bench_generalist_map.png)
*Figure 5: Scatter plot mapping Training Efficiency vs. HPO Accuracy. Elite generalist algorithms occupy the top-right quadrant.*

---

## 📁 Repository Structure

- `swarmtorch/`: The root package.
  - `base/`: Core abstract classes and shared logic for all optimizers.
  - `swarm/`, `evolutionary/`, `physics/`, etc.: Implementation of ~120 algorithms.
  - **`benchmarks/`**: A self-contained research folder containing:
    - `master_benchmark.py`: The full experimental suite.
    - `COMPREHENSIVE_EXPERIMENT_REPORT.md`: Detailed per-algorithm results.
    - All high-resolution visualization assets (.png).
- `tests/`: Automated unit tests for stability and regression.

---

## 🤝 Acknowledgments & References

This library was developed with reference to the **[pyMetaheuristic](https://github.com/mariosv/pyMetaheuristic)** library. We are grateful for their contributions to the metaheuristic optimization community, which served as a foundational resource for the algorithmic implementations in **SwarmTorch**.

## 📝 Citation

If you use **SwarmTorch** in your research, please cite it as follows:

```bibtex
@software{swarmtorch2026,
  author = {Halleluyah Darasimi Oludele},
  title = {SwarmTorch: A PyTorch Library for 120+ Metaheuristic Optimization Algorithms},
  year = {2026},
  url = {https://github.com/hallelx2/swarmtorch}
}
```

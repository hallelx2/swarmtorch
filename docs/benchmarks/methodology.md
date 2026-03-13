# Benchmark Methodology

This document describes how SwarmTorch benchmarks were conducted.

## Overview

We evaluated all 120 algorithms (60 model-training + 60 HPO searchers) on standardized tasks to compare performance.

## Model Training Benchmarks

### Task: XOR Classification

**Problem**: 2-layer MLP for XOR classification

```python
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
```

**Dataset**: 4 samples
- Input: `[[0,0], [0,1], [1,0], [1,1]]`
- Output: `[[0], [1], [1], [0]]`

**Loss Function**: BCEWithLogitsLoss

**Budget**: 1000 forward evaluations per optimizer

### Metrics

- **Final Loss**: BCE loss after budget exhausted
- **Test Accuracy**: Classification accuracy on test data

### Methodology

```python
eval_count = 0
BUDGET = 1000

while eval_count < BUDGET:
    def closure():
        nonlocal eval_count
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        eval_count += 1  # Count forward passes
        return loss
    
    optimizer.step(closure)
```

<Note>
We count forward passes (not optimizer steps) because swarm optimizers evaluate multiple candidates per step.
</Note>

---

## HPO Benchmarks

### Task: Binary Classification

**Problem**: MLP for synthetic binary classification

```python
model = nn.Sequential(
    nn.Linear(10, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1)
)
```

**Dataset**: 100 samples, 10 features, binary labels

**Search Space**:
```python
param_space = {
    'lr': (0.001, 0.1),        # Continuous
    'hidden_dim': [16, 32, 64], # Categorical
    'dropout': (0.0, 0.5),      # Continuous
}
```

**Budget**: 2 iterations × 10 swarm_size = ~20 evaluations per searcher

### Metrics

- **Validation Accuracy**: Classification accuracy on held-out data
- **Success Rate**: % of runs achieving >90% accuracy

---

## Algorithms Tested

### Model Training (60)

| Category | Count | Algorithms |
|----------|-------|------------|
| Swarm | 32 | PSO, GWO, WOA, HHO, SSA, Firefly, Bat, Dragonfly, CuckooSearch, Salp, Bee, Fish, DFO, MBO, CSA, AOA, SOS, DVBA, ABCO, GOA, HUS, JY, SPBO, RandomSearch, IGWO, IWOA, ACGWO, Memetic, Clonalg, AFSA, HSA, DFO2 |
| Evolutionary | 8 | DE, GA, CEM, PFA, ARS, FDA, CA, PBIL |
| Physics | 3 | SA, GSA, FPA |
| Bio-Inspired | 5 | ALO, BBO, MVO, SineCosine, MFO |
| Human-Based | 2 | TLBO, HarmonySearch |
| Hybrid | 10 | EHO, ChickenSwarm, SMA, CatSwarm, Cockroach, Coati, Gorilla, GorillaTroopsOptimizer, JSO, KHA |
| **Total** | **60** | |

### HPO Searchers (60)

All 60 model-training optimizers have corresponding HPO searchers via `GenericSwarmSearch`.

---

## Reproducibility

Benchmarks use fixed random seeds for reproducibility:

```python
torch.manual_seed(42)
np.random.seed(42)
```

---

## Running Benchmarks

Install benchmark dependencies:

```bash
pip install swarmtorch[benchmarks]
```

Run benchmark suite:

```python
from swarmtorch.benchmarks import run_benchmarks

results = run_benchmarks()
```

For more details, see the benchmarks source code in `benchmarks/` directory.

# Choosing an Algorithm

With 60 algorithms available, choosing the right one can be overwhelming. This guide helps you decide.

## Decision Flowchart

```
Is your problem differentiable?
├─ YES → Consider standard optimizers (Adam/SGD) first
│         SwarmTorch as fallback for non-differentiable cases
│
└─ NO → Continue below

What are you optimizing?
├─ Model WEIGHTS → Model Training Optimizers
└─ Model HYPERPARAMETERS → HPO Searchers

How complex is your search space?
├─ Simple (few params, convex) → PSO, DE
├─ Complex (many params, multi-modal) → GWO, WOA, HHO
└─ Very complex (discrete + continuous) → GA, CEM

How much compute budget?
├─ Limited → PSO (fast convergence)
├─ Moderate → GWO, DE (balanced)
└─ Large → HHO, SSA (thorough exploration)
```

## Algorithm Recommendations by Use Case

### Model Training

| Use Case | First Choice | Alternatives |
|----------|--------------|--------------|
| **General purpose** | PSO | GWO, DE |
| **Multi-modal landscapes** | GWO | WOA, HHO |
| **Fast convergence needed** | PSO | DE, GA |
| **Complex non-convex** | WOA | HHO, SSA |
| **Evolutionary preference** | DE | GA, CEM |
| **Simulated annealing** | SA | - |
| **Large population exploration** | GA | DE |

### Hyperparameter Tuning

| Use Case | First Choice | Alternatives |
|----------|--------------|--------------|
| **Quick baseline** | RandomSearchHT | - |
| **General HPO** | PSOSearch | GWOSearch |
| **Continuous space** | WOASearch | HHOSearch |
| **Mixed discrete + continuous** | DESearch | GASearch |
| **Large search space** | GWOSearch | WOA |

## Algorithm Characteristics

### Swarm Intelligence (32 optimizers)

| Algorithm | Convergence | Exploration | Best For |
|-----------|-------------|-------------|----------|
| **PSO** | Fast | Medium | General purpose, fast results |
| **GWO** | Medium | High | Complex multi-modal problems |
| **WOA** | Medium | High | Whale-inspired, balanced |
| **HHO** | Medium | Very High | Harris hawk hunting |
| **SSA** | Medium | High | Sparrow foraging |
| **Firefly** | Slow | Very High | Light-attraction behavior |
| **Bat** | Medium | High | Echolocation |
| **Dragonfly** | Medium | High | Dynamic swarms |

### Evolutionary (8 optimizers)

| Algorithm | Convergence | Exploration | Best For |
|-----------|-------------|-------------|----------|
| **DE** | Fast | Medium | Continuous optimization |
| **GA** | Medium | High | Discrete + continuous |
| **CEM** | Fast | Low | Gaussian-based |
| **PBIL** | Fast | Low | Probability-based |

### Physics-Based (3 optimizers)

| Algorithm | Convergence | Exploration | Best For |
|-----------|-------------|-------------|----------|
| **SA** | Slow | Very High | Temperature-based |
| **GSA** | Medium | High | Gravity-based |
| **FPA** | Medium | High | Flower pollination |

### Bio-Inspired (5 optimizers)

| Algorithm | Best For |
|-----------|----------|
| **ALO** | Ant lion hunting |
| **BBO** | Biogeography |
| **MVO** | Multi-verse optimization |
| **SineCosine** | Sine-cosine waves |
| **MFO** | Moth-flame optimization |

### Human-Based (2 optimizers)

| Algorithm | Best For |
|-----------|----------|
| **TLBO** | Teaching-learning |
| **HarmonySearch** | Music improvisation |

### Hybrid (10 optimizers)

| Algorithm | Best For |
|-----------|----------|
| **SMA** | Slime mold |
| **Gorilla** | Gorilla troop behavior |
| **EHO** | Elephant herding |
| **JSO** | Jaya optimization |

## Benchmark Results Summary

Based on our comprehensive benchmarks:

### Model Training (Lower is Better)

| Rank | Algorithm | Loss |
|------|-----------|------|
| 1 | CA | 0.1317 |
| 2 | HHO | 0.1668 |
| 3 | CEM | 0.1995 |
| 4 | SSA | 0.2201 |
| 5 | GWO | 0.2462 |

### HPO Searchers (Higher Accuracy is Better)

| Rank | Searcher | Accuracy |
|------|----------|----------|
| 1 | SASearch | 98.5% |
| 2 | DVBASearch | 98.5% |
| 3 | PBILSearch | 98.0% |
| 4 | CASearch | 97.9% |
| 5 | HHOSearch | 97.5% |

<Note>
Benchmark results may vary based on problem characteristics. We recommend trying multiple algorithms.
</Note>

## Quick Selection Guide

```python
# Start here based on your problem:

# 1. Quick test
from swarmtorch import PSO

# 2. Multi-modal / complex
from swarmtorch import GWO, WOA, HHO

# 3. Need exploration
from swarmtorch import DE, GA

# 4. For HPO
from swarmtorch import PSOSearch, GWOSearch
```

## Tips for Selection

1. **Start Simple**: Try PSO first for model training, PSOSearch for HPO
2. **If Stuck**: Move to GWO or WOA for model training
3. **Compare**: Run 2-3 algorithms and pick the best
4. **Ensemble**: Use different algorithms for initialization

## When NOT to Use SwarmTorch

- **Standard differentiable problems**: Use Adam/SGD
- **Very large models**: Gradient-based is more efficient
- **Time-critical applications**: SwarmTorch has more function evaluations
- **Convex problems**: Standard optimizers converge faster

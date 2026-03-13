# Evolutionary Algorithms

Evolutionary algorithms mimic natural selection and genetic processes to optimize model parameters and hyperparameters.

## Model Training Optimizers

### DE

Differential Evolution optimizer. An evolutionary algorithm that optimizes a problem by iteratively trying to improve a candidate solution using vector differences to perturb the population.

```python
from swarmtorch.evolutionary import DE
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `population_size` | int | 30 | Number of individuals in the population |
| `cr` | float | 0.9 | Crossover rate (probability of crossover) |
| `f` | float | 0.8 | Mutation factor (controls amplification of difference vectors) |
| `device` | str | "cpu" | Device for computation |

#### Example

```python
optimizer = DE(model.parameters(), population_size=30, cr=0.9, f=0.8)
```

---

### GA

Genetic Algorithm optimizer. An evolutionary algorithm that simulates natural selection using selection, crossover, and mutation operators.

```python
from swarmtorch.evolutionary import GA
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `population_size` | int | 30 | Number of individuals in population |
| `crossover_rate` | float | 0.9 | Probability of crossover |
| `mutation_rate` | float | 0.1 | Probability of mutation |
| `device` | str | "cpu" | Device for computation |

#### Example

```python
optimizer = GA(model.parameters(), population_size=30, crossover_rate=0.9, mutation_rate=0.1)
```

---

### CEM

Cross-Entropy Method optimizer. Uses statistical sampling to estimate and optimize the distribution of candidate solutions.

```python
from swarmtorch.evolutionary import CEM
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `population_size` | int | 30 | Number of samples in population |
| `device` | str | "cpu" | Device for computation |

#### Example

```python
optimizer = CEM(model.parameters(), population_size=30)
```

---

### PFA

Peacock Algorithm optimizer. Inspired by peacock mating behavior, uses positional updates based on the best position with decreasing exploration.

```python
from swarmtorch.evolutionary import PFA
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `swarm_size` | int | 30 | Number of peacocks in the swarm |
| `device` | str | "cpu" | Device for computation |

#### Example

```python
optimizer = PFA(model.parameters(), swarm_size=30)
```

---

### ARS

Archimedes Optimization Algorithm optimizer. Based on physical laws of buoyancy and density, uses inter-particle forces for exploration.

```python
from swarmtorch.evolutionary import ARS
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `swarm_size` | int | 30 | Number of particles in the swarm |
| `device` | str | "cpu" | Device for computation |

#### Example

```python
optimizer = ARS(model.parameters(), swarm_size=30)
```

---

### FDA

Forest Defense Algorithm optimizer. Inspired by defense mechanisms in forest ecosystems, uses collective behavior for optimization.

```python
from swarmtorch.evolutionary import FDA
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `swarm_size` | int | 30 | Number of trees in the forest |
| `device` | str | "cpu" | Device for computation |

#### Example

```python
optimizer = FDA(model.parameters(), swarm_size=30)
```

---

### CA

Cultural Algorithm optimizer. Incorporates a belief space that stores knowledge gained during the search process to guide population evolution.

```python
from swarmtorch.evolutionary import CA
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `swarm_size` | int | 30 | Number of individuals in population |
| `device` | str | "cpu" | Device for computation |

#### Example

```python
optimizer = CA(model.parameters(), swarm_size=30)
```

---

### PBIL

Population-Based Incremental Learning optimizer. Combines evolutionary strategies with incremental learning by maintaining a probability vector.

```python
from swarmtorch.evolutionary import PBIL
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | iterable | Required | Model parameters to optimize |
| `population_size` | int | 30 | Number of individuals in population |
| `device` | str | "cpu" | Device for computation |

#### Example

```python
optimizer = PBIL(model.parameters(), population_size=30)
```

---

## HPO Searchers

All HPO searchers inherit from `GenericSwarmSearch` and wrap the corresponding optimizer for hyperparameter tuning.

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_fn` | callable | Required | Function that builds a model from hyperparameters |
| `param_space` | dict | Required | Hyperparameter search space |
| `train_fn` | callable | Required | Training function that returns loss |
| `iterations` | int | 50 | Number of search iterations |
| `swarm_size` | int | 30 | Number of candidates per iteration |
| `device` | str | "cpu" | Device for computation |
| `verbose` | bool | True | Whether to print progress |

#### Parameter Space Format

```python
param_space = {
    'lr': (0.001, 0.1),           # Continuous: (min, max)
    'hidden_dim': [16, 32, 64],    # Categorical: list of choices
}
```

---

### DESearch

Hyperparameter search using Differential Evolution.

```python
from swarmtorch.evolutionary import DESearch
```

#### Example

```python
searcher = DESearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'batch_size': [16, 32, 64]},
    train_fn=train_model,
    iterations=50,
)
best_params = searcher.search()
```

---

### GASearch

Hyperparameter search using Genetic Algorithm.

```python
from swarmtorch.evolutionary import GASearch
```

#### Example

```python
searcher = GASearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'batch_size': [16, 32, 64]},
    train_fn=train_model,
    iterations=50,
)
best_params = searcher.search()
```

---

### CEMSearch

Hyperparameter search using Cross-Entropy Method.

```python
from swarmtorch.evolutionary import CEMSearch
```

#### Example

```python
searcher = CEMSearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'batch_size': [16, 32, 64]},
    train_fn=train_model,
    iterations=50,
)
best_params = searcher.search()
```

---

### PFASearch

Hyperparameter search using Peacock Algorithm.

```python
from swarmtorch.evolutionary import PFASearch
```

#### Example

```python
searcher = PFASearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'batch_size': [16, 32, 64]},
    train_fn=train_model,
    iterations=50,
)
best_params = searcher.search()
```

---

### ARSSearch

Hyperparameter search using Archimedes Optimization Algorithm.

```python
from swarmtorch.evolutionary import ARSSearch
```

#### Example

```python
searcher = ARSSearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'batch_size': [16, 32, 64]},
    train_fn=train_model,
    iterations=50,
)
best_params = searcher.search()
```

---

### FDASearch

Hyperparameter search using Forest Defense Algorithm.

```python
from swarmtorch.evolutionary import FDASearch
```

#### Example

```python
searcher = FDASearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'batch_size': [16, 32, 64]},
    train_fn=train_model,
    iterations=50,
)
best_params = searcher.search()
```

---

### CASearch

Hyperparameter search using Cultural Algorithm.

```python
from swarmtorch.evolutionary import CASearch
```

#### Example

```python
searcher = CASearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'batch_size': [16, 32, 64]},
    train_fn=train_model,
    iterations=50,
)
best_params = searcher.search()
```

---

### PBILSearch

Hyperparameter search using Population-Based Incremental Learning.

```python
from swarmtorch.evolutionary import PBILSearch
```

#### Example

```python
searcher = PBILSearch(
    model_fn=build_model,
    param_space={'lr': (0.001, 0.1), 'batch_size': [16, 32, 64]},
    train_fn=train_model,
    iterations=50,
)
best_params = searcher.search()
```

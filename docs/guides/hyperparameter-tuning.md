# Hyperparameter Tuning

SwarmTorch provides 60 hyperparameter optimization (HPO) searchers. This guide shows how to use them effectively.

## How It Works

The key insight: **any model-training optimizer can be used for HPO** via the `GenericSwarmSearch` adapter:

1. Create a "dummy model" with parameters in [0,1] space
2. Encode hyperparameters as model weights
3. The swarm optimizer "trains" these weights
4. Decode learned weights back to hyperparameters

## Basic Usage

### Step 1: Define Your Model Factory

```python
import torch.nn as nn

def build_model(hidden_dim: int) -> nn.Module:
    """Factory function that builds your model."""
    return nn.Sequential(
        nn.Linear(4, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )
```

### Step 2: Define Training Function

```python
def train_fn(model: nn.Module) -> float:
    """Train model briefly and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Quick training
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    
    # Return validation loss (minimize)
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()
    return val_loss
```

### Step 3: Define Search Space

```python
param_space = {
    'lr': (0.001, 0.1),           # Continuous: (min, max)
    'hidden_dim': [16, 32, 64, 128],  # Categorical: list of choices
    'dropout': (0.0, 0.5),        # Continuous range
}
```

### Step 4: Run Search

```python
from swarmtorch import PSOSearch

searcher = PSOSearch(
    model_fn=build_model,
    param_space=param_space,
    train_fn=train_fn,
    iterations=30,
    swarm_size=20,
)

best_params = searcher.search()
print(f"Best params: {best_params}")
```

## Parameter Space Definition

### Continuous Parameters

```python
'lr': (0.001, 0.1)  # Searches range [0.001, 0.1]
'beta': (-1.0, 1.0)  # Any float range
```

### Categorical Parameters

```python
'optimizer': ['adam', 'sgd', 'rmsprop']  # Discrete choices
'hidden_dim': [32, 64, 128, 256]        # Any type allowed
```

### Mixed Spaces

```python
param_space = {
    'lr': (0.001, 0.1),           # Continuous
    'hidden_dim': [16, 32, 64],    # Categorical
    'layers': [2, 3, 4],           # Categorical (int)
    'dropout': (0.0, 0.5),        # Continuous
}
```

## Available Searchers

All 60 model-training optimizers have corresponding HPO searchers:

### Swarm Intelligence Searchers (32)

```python
from swarmtorch import (
    PSOSearch, GWOSearch, WOASearch, HHOSearch, SSASearch,
    FireflySearch, BatSearch, DragonflySearch, CuckooSearchHT,
    SalpSearch, BeeSearch, FishSearch, DFOSearch, MBOSearch,
    CSASearch, AOASearch, SOSSearch, DVBASearch, ABCOSearch,
    GOASearch, HUSSearch, JYSearch, SPBOSearch, RandomSearchHT,
    IGWOSearch, IWOASearch, ACGWOSearch, MemeticSearch, ClonalgSearch,
    AFSASearch, HSASearch, DFO2Search,
)
```

### Evolutionary Searchers (8)

```python
from swarmtorch import (
    DESearch, GASearch, CEMSearch, PFASearch, ARSSearch,
    FDASearch, CASearch, PBILSearch,
)
```

### Physics-Based Searchers (3)

```python
from swarmtorch import SASearch, GSASearch, FPASearch
```

### Bio-Inspired Searchers (5)

```python
from swarmtorch import (
    ALOSearch, BBOSearch, MVOSearch, SineCosineSearch, MFOSearch,
)
```

### Human-Based Searchers (2)

```python
from swarmtorch import TLBOSearch, HarmonySearchHT
```

### Hybrid Searchers (10)

```python
from swarmtorch import (
    EHOSearch, ChickenSwarmSearch, SMASearch, CatSwarmSearch,
    CockroachSearch, CoatiSearch, GorillaSearch, GorillaTroopsOptimizerSearch,
    JSOSearch, KHASearch,
)
```

## Searcher Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_fn` | callable | Required | Function that builds model from params |
| `param_space` | dict | Required | Hyperparameter search space |
| `train_fn` | callable | Required | Function that trains model and returns loss |
| `iterations` | int | 50 | Number of search iterations |
| `swarm_size` | int | 10 | Candidates per iteration |
| `device` | str | "cpu" | Device for computation |
| `verbose` | bool | True | Print progress |

## Complete Example: CNN Hyperparameter Search

```python
import torch
import torch.nn as nn
from swarmtorch import GWOSearch

# Dataset
X = torch.rand(100, 1, 28, 28)  # MNIST-like
y = torch.randint(0, 10, (100,))

def build_model(conv_channels: int, kernel_size: int, lr: float):
    return nn.Sequential(
        nn.Conv2d(1, conv_channels, kernel_size),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(conv_channels * 12 * 12, 10)
    )

def train_fn(model):
    opt = torch.optim.Adam(model.parameters(), lr=model.hparams['lr'])
    crit = nn.CrossEntropyLoss()
    
    for _ in range(10):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
    
    with torch.no_grad():
        return crit(model(X), y).item()

searcher = GWOSearch(
    model_fn=build_model,
    param_space={
        'conv_channels': [16, 32, 64],
        'kernel_size': [3, 5],
        'lr': (0.001, 0.01),
    },
    train_fn=train_fn,
    iterations=20,
    swarm_size=15,
)

best = searcher.search()
print(best)
```

## Best Practices

### 1. Start with Fewer Iterations

```python
# Quick search
searcher = PSOSearch(..., iterations=10, swarm_size=10)

# Refined search
searcher = PSOSearch(..., iterations=50, swarm_size=20)
```

### 2. Balance Exploration vs Exploitation

| Searcher | Best For |
|----------|----------|
| PSOSearch | Balanced exploration/exploitation |
| GWOSearch | Exploitation-heavy |
| WOASearch | Balanced |
| DESearch | Exploration-heavy |
| RandomSearchHT | Baseline comparison |

### 3. Use Proper Validation

```python
def train_fn(model):
    # Train on train split
    train_loss = run_training(model, train_data)
    
    # Evaluate on validation
    val_loss = evaluate(model, val_data)
    
    return val_loss  # Return validation loss
```

### 4. Set Seeds for Reproducibility

```python
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
```

## Troubleshooting

### Search Not Converging

- Increase `iterations`
- Increase `swarm_size`
- Try different searcher (PSO → GWO → WOA)

### Best Params at Boundary

- Your search space may be too narrow
- Expand the range

### Training Too Slow

- Reduce `iterations` for initial search
- Reduce training epochs in `train_fn`
- Use GPU: `device="cuda"`

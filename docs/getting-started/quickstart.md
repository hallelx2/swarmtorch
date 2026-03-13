# Quickstart

This guide will get you running with SwarmTorch in 5 minutes. We'll cover both **model training** and **hyperparameter optimization**.

## 1. Gradient-Free Model Training

SwarmTorch optimizers work like standard PyTorch optimizers but use swarm intelligence instead of gradients.

### Example: XOR Classification

```python
import torch
import torch.nn as nn
from swarmtorch import PSO

# Define a simple neural network
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# Create PSO optimizer (Particle Swarm Optimization)
optimizer = PSO(model.parameters(), swarm_size=30)

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Training loop - swarm optimizers require a closure
for epoch in range(200):
    def closure():
        optimizer.zero_grad()
        output = model(X)
        # Use BCEWithLogitsLoss for metaheuristic training
        loss = nn.BCEWithLogitsLoss()(output, y)
        return loss
    
    loss = optimizer.step(closure)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Test predictions
with torch.no_grad():
    predictions = (torch.sigmoid(model(X)) > 0.5).float()
    print(f"Predictions: {predictions.T}")
    print(f"Expected:    {y.T}")
```

### Key Differences from Standard Optimizers

| Standard (Adam/SGD) | SwarmTorch |
|---------------------|------------|
| Uses gradients | Uses loss values only |
| Single solution per step | Evaluates multiple candidates per step |
| `optimizer.step()` | `optimizer.step(closure)` (closure required) |

## 2. Hyperparameter Optimization

Use SwarmTorch searchers to find optimal hyperparameters for your models.

### Example: Tuning Learning Rate and Hidden Dimensions

```python
import torch
import torch.nn as nn
from swarmtorch import PSOSearch

def build_model(hidden_dim: int) -> nn.Module:
    """Build a simple model with configurable hidden dimension."""
    return nn.Sequential(
        nn.Linear(4, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

def train_model(model: nn.Module, lr: float, X: torch.Tensor, y: torch.Tensor) -> float:
    """Train the model and return validation loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X), y).item()
    return val_loss

# Generate some data
torch.manual_seed(42)
X = torch.rand(100, 4)
y = (X.sum(dim=1) > 2).float().unsqueeze(1)

# Define hyperparameter search space
param_space = {
    'lr': (0.001, 0.1),           # Continuous: (min, max)
    'hidden_dim': [16, 32, 64],    # Categorical: discrete choices
}

# Run PSO hyperparameter search
searcher = PSOSearch(
    model_fn=build_model,
    param_space=param_space,
    train_fn=lambda model: train_model(model, model.hparams['lr'], X, y),
    iterations=20,
    swarm_size=10,
)

best_params = searcher.search()
print(f"Best hyperparameters: {best_params}")
```

## 3. Choosing an Algorithm

Not sure which algorithm to use? Here's a quick guide:

### For Model Training

| Scenario | Recommended Algorithm |
|----------|---------------------|
| General purpose, fast convergence | **PSO** |
| Complex, multi-modal landscapes | **GWO**, **WOA** |
| Exploitation-heavy problems | **HHO**, **SSA** |
| Evolutionary approach preferred | **DE**, **GA** |
| Simulated annealing needed | **SA** |

### For Hyperparameter Tuning

| Scenario | Recommended Searcher |
|----------|---------------------|
| General purpose | **PSOSearch**, **GWOSearch** |
| Exploration-heavy | **WOASearch**, **HHOSearch** |
| Large search space | **DESearch**, **GASearch** |
| Quick baseline | **RandomSearchHT** |

## 4. Next Steps

- Read the [Model Training Guide](/guides/model-training) for deeper dive
- Learn about [Hyperparameter Tuning](/guides/hyperparameter-tuning)
- Explore the [Choosing an Algorithm](/guides/choosing-algorithm) guide
- Check out [Examples](/examples) for real-world use cases
